use tensorflow::Tensor;

use features::Layer;
use guide::{BatchGuide, Guide};
use models::tensorflow::{InstanceSlices, LayerTensors, TensorflowModel};
use system::{ParserState, TransitionSystem};

impl<T> Guide for TensorflowModel<T>
where
    T: TransitionSystem,
{
    type Transition = T::Transition;

    fn best_transition(&mut self, state: &ParserState) -> Self::Transition {
        self.best_transitions(&[state]).remove(0)
    }
}

impl<T> BatchGuide for TensorflowModel<T>
where
    T: TransitionSystem,
{
    type Transition = T::Transition;

    fn best_transitions(&mut self, states: &[&ParserState]) -> Vec<Self::Transition> {
        if states.is_empty() {
            return Vec::new();
        }

        // Allocate batch tensors.
        let embed_size = self.vectorizer().embedding_layer_size();
        let mut embed_tensors = Tensor::new(&[states.len() as u64, embed_size as u64]);

        let mut input_lookup_tensors = LayerTensors::new();
        for (layer, size) in self.vectorizer().lookup_layer_sizes() {
            input_lookup_tensors[layer] = Tensor::new(&[states.len() as u64, size as u64]).into();
        }

        let n_deprel_embeds = self
            .vectorizer()
            .layer_lookups()
            .layer_lookup(Layer::DepRel)
            .unwrap()
            .len();
        let n_non_lookup_inputs = n_deprel_embeds * T::ATTACHMENT_ADDRS.len();
        let mut input_non_lookup_tensors =
            Tensor::new(&[states.len() as u64, n_non_lookup_inputs as u64, ]);

        // Fill tensors.
        for (idx, state) in states.iter().enumerate() {
            let embed_offset = embed_size * idx;
            self.vectorizer().realize_into(
                state,
                &mut embed_tensors[embed_offset..embed_offset + embed_size],
                &mut input_lookup_tensors.to_instance_slices(idx),
                &mut input_non_lookup_tensors[(idx * n_non_lookup_inputs)
                    ..(idx * n_non_lookup_inputs + n_non_lookup_inputs)],
                &T::ATTACHMENT_ADDRS,
            );
        }

        self.predict(states, &embed_tensors, &input_lookup_tensors, &input_non_lookup_tensors)
    }
}

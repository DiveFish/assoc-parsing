use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::io::BufRead;
use std::result;

use enum_map::EnumMap;
use failure::Error;
use tensorflow::Tensor;

use features::addr;
use features::lookup::LookupResult;
use features::parse_addr::parse_addressed_values;
use features::{BoxedLookup, Lookup, LookupType};
use system::{AttachmentAddr,ParserState};

/// Multiple addressable parts of the parser state.
///
/// `AddressedValues` represents multiple addresses into the parser state.
/// This can be used to construct feature vectors over the parser state.
pub struct AddressedValues(pub Vec<addr::AddressedValue>);

impl AddressedValues {
    /// Read addressed values specification from a text file.
    ///
    /// Such a text file consists of lines with the format
    ///
    /// ~~~text,no_run
    /// [address+] layer
    /// ~~~
    ///
    /// Multiple addresses are used to e.g. address the left/rightmost
    /// dependency of a token on the stack or buffer.
    pub fn from_buf_read<R>(mut read: R) -> Result<Self, Error>
    where
        R: BufRead,
    {
        let mut data = String::new();
        read.read_to_string(&mut data)?;
        Ok(AddressedValues(parse_addressed_values(&data)?))
    }
}

/// A feature vector.
///
/// `InputVector` instances represent feature vectors, also called
/// input layers in neural networks. The input vector is split in
/// vectors for different layers. In each layer, the feature is encoded
/// as a 32-bit identifier, which is typically the row of the layer
/// value in an embedding matrix. The second type of layer directly
/// stores floating point values which can represent, for example,
/// association measures.
pub struct InputVector {
    pub lookup_layers: EnumMap<Layer, Vec<i32>>,
    pub non_lookup_layer: Vec<f32>,
    pub embed_layer: Tensor<f32>,
}

#[derive(Clone, Copy, Debug, Enum, Eq, PartialEq)]
pub enum Layer {
    Token,
    Tag,
    DepRel,
    Feature,
}

impl fmt::Display for Layer {
    fn fmt(&self, f: &mut fmt::Formatter) -> result::Result<(), fmt::Error> {
        let s = match *self {
            Layer::Token => "tokens",
            Layer::Tag => "tags",
            Layer::DepRel => "deprels",
            Layer::Feature => "features",
        };

        f.write_str(s)
    }
}

// I am not sure whether I like the use of Borrow here, but is there another
// convenient way to convert from both addr::Layer and &addr::Layer?
impl From<&addr::Layer> for Layer {
    fn from(layer: &addr::Layer) -> Self {
        match layer {
            &addr::Layer::Token => Layer::Token,
            &addr::Layer::Tag => Layer::Tag,
            &addr::Layer::DepRel => Layer::DepRel,
            &addr::Layer::Feature(_) => Layer::Feature,
        }
    }
}

/// Lookups for layers.
///
/// This data structure bundles lookups for the different layers (tokens,
/// part-of-speech, etc).
pub struct LayerLookups(EnumMap<Layer, BoxedLookup>);

impl LayerLookups {
    pub fn new() -> Self {
        LayerLookups(EnumMap::new())
    }

    pub fn insert<L>(&mut self, layer: Layer, lookup: L)
    where
        L: Into<Box<Lookup>>,
    {
        self.0[layer] = BoxedLookup::new(lookup)
    }

    /// Get the lookup for a layer.
    pub fn layer_lookup(&self, layer: Layer) -> Option<&Lookup> {
        self.0[layer].as_ref()
    }
}

/// Vectorizer for parser states.
///
/// An `InputVectorizer` vectorizes parser states.
pub struct InputVectorizer {
    layer_lookups: LayerLookups,
    input_layer_addrs: AddressedValues,
    association_strengths: HashMap<(String, String, String), f32>,
    no_lowercase_tags: Vec<String>,
}

impl InputVectorizer {
    /// Construct an input vectorizer.
    ///
    /// The vectorizer is constructed from the layer lookups and the parser
    /// state addresses from which the feature vector should be used. The layer
    /// lookups are used to find the indices that represent the features.
    pub fn new(layer_lookups: LayerLookups, input_addrs: AddressedValues, association_strengths: HashMap<(String, String, String), f32>, no_lowercase_tags: Vec<String>) -> Self {
        InputVectorizer {
            layer_lookups: layer_lookups,
            input_layer_addrs: input_addrs,
            association_strengths,
            no_lowercase_tags,
        }
    }

    pub fn embedding_layer_size(&self) -> usize {
        let mut size = 0;

        for layer in &self.input_layer_addrs.0 {
            if let Some(lookup) = self.layer_lookups.0[(&layer.layer).into()].as_ref() {
                match lookup.lookup_type() {
                    LookupType::Embedding(dims) => size += dims,
                    LookupType::Index => (),
                }
            }
        }

        size
    }

    pub fn layer_addrs(&self) -> &AddressedValues {
        &self.input_layer_addrs
    }

    /// Get the layer lookups.
    pub fn layer_lookups(&self) -> &LayerLookups {
        &self.layer_lookups
    }

    pub fn association_strengths(&self) -> &HashMap<(String, String, String), f32> {
        &self.association_strengths
    }

    pub fn lookup_layer_sizes(&self) -> EnumMap<Layer, usize> {
        let mut sizes = EnumMap::new();

        for layer in &self.input_layer_addrs.0 {
            if let Some(lookup) = self.layer_lookups.0[(&layer.layer).into()].as_ref() {
                match lookup.lookup_type() {
                    LookupType::Embedding(_) => (),
                    LookupType::Index => sizes[(&layer.layer).into()] += 1,
                }
            }
        }

        sizes
    }

    /// Vectorize a parser state.
    pub fn realize(&self, state: &ParserState, attachment_addrs: &[AttachmentAddr]) -> InputVector {
        let mut embed_layer = Tensor::new(&[self.embedding_layer_size() as u64]);

        let mut lookup_layers = EnumMap::new();
        for (layer, &size) in &self.lookup_layer_sizes() {
            lookup_layers[layer] = vec![0; size];
        }

        let n_deprel_embeds = &self
            .layer_lookups
            .layer_lookup(Layer::DepRel)
            .unwrap()
            .len();
        let mut non_lookup_layer = vec![0f32; n_deprel_embeds * attachment_addrs.len()];

        self.realize_into(
            state,
            &mut embed_layer,
            &mut lookup_layers,
            &mut non_lookup_layer,
            attachment_addrs,
        );

        InputVector {
            lookup_layers,
            non_lookup_layer,
            embed_layer,
        }
    }
    pub fn realize_into<S>(
        &self,
        state: &ParserState,
        embed_layer: &mut [f32],
        lookup_slices: &mut EnumMap<Layer, S>,
        non_lookup_slice: &mut [f32],
        attachment_addrs: &[AttachmentAddr],
    ) where
        S: AsMut<[i32]>,
    {
        self.realize_into_lookups(state, embed_layer, lookup_slices);
        self.realize_into_assoc_strengths(state, non_lookup_slice, attachment_addrs);
    }

    /// Vectorize a parser state into the given slices.
    pub fn realize_into_lookups<S>(
        &self,
        state: &ParserState,
        embed_layer: &mut [f32],
        lookup_slices: &mut EnumMap<Layer, S>,
    ) where
        S: AsMut<[i32]>,
    {
        let mut embed_offset = 0;
        let mut layer_offsets: EnumMap<Layer, usize> = EnumMap::new();

        for layer in &self.input_layer_addrs.0 {
            let val = layer.get(state);
            let mut offset = &mut layer_offsets[(&layer.layer).into()];

            let layer = &layer.layer;

            match lookup_value(
                self.layer_lookups
                    .layer_lookup(layer.into())
                    .expect(&format!("Missing layer lookup for: {:?}", layer)),
                val,
            ) {
                LookupResult::Embedding(embed) => {
                    embed_layer[embed_offset..embed_offset + embed.len()]
                        .copy_from_slice(embed.as_slice().expect("Embedding is not contiguous"));
                    embed_offset += embed.len();
                }
                LookupResult::Index(idx) => {
                    lookup_slices[layer.into()].as_mut()[*offset] = idx as i32;
                    *offset += 1;
                }
            }
        }
    }
    /// Vectorize a parser state into the given association measure slices.
    ///
    /// Add to `non_lookup_slice` the association measure between all parser state addresses
    /// undergoing attachment. Consider all possible dependency relations.
    pub fn realize_into_assoc_strengths(
        &self,
        state: &ParserState,
        non_lookup_slice: &mut [f32],
        attachment_addrs: &[AttachmentAddr],
    ) {
        if let Some(deprel_layer) = self.layer_lookups.layer_lookup(Layer::DepRel) {
            let deprels = deprel_layer.lookup_values();

            for (idx, (addr, deprel)) in
                iproduct!(attachment_addrs.iter(), deprels.iter()).enumerate()
                {
                    let addr_head = addr::AddressedValue {
                        address: vec![addr.head],
                        layer: addr::Layer::Token,
                    };
                    let addr_dependent = addr::AddressedValue {
                        address: vec![addr.dependent],
                        layer: addr::Layer::Token,
                    };
                    let addr_head_pos = addr::AddressedValue {
                        address: vec![addr.head],
                        layer: addr::Layer::Tag,
                    };
                    let addr_dependent_pos = addr::AddressedValue {
                        address: vec![addr.dependent],
                        layer: addr::Layer::Tag,
                    };
                    let head = addr_head.get(state);
                    let dependent = addr_dependent.get(state);
                    let head_pos = addr_head_pos.get(state);
                    let dependent_pos = addr_dependent_pos.get(state);

                    if let (Some(head), Some(dependent), Some(head_pos), Some(dependent_pos)) =
                    (head, dependent, head_pos, dependent_pos) {
                        let association = self.assoc_strength(&head, &dependent, &head_pos, &dependent_pos, &deprel);
                        non_lookup_slice[idx] = association;
                    }
                }
        }
    }

    fn assoc_strength(
        &self,
        head: &str,
        dependent: &str,
        head_pos: &str,
        dependent_pos: &str,
        deprel: &str,
    ) -> f32 {
        let mut head = head.to_string();
        let mut dependent = dependent.to_string();

        if !self.no_lowercase_tags.contains(&head_pos.to_string()) {
            head = head.to_lowercase();
        }
        if !self.no_lowercase_tags.contains(&dependent_pos.to_string()) {
            dependent = dependent.to_lowercase();
        }
        let dep_triple = (head.to_string(), dependent.to_string(), deprel.to_string());
        match self.association_strengths.get(&dep_triple) {
            Some(association_strength) => *association_strength,
            None => 0.0,
        }
    }
}

fn lookup_value<'a>(lookup: &'a Lookup, feature: Option<Cow<str>>) -> LookupResult<'a> {
    match feature {
        Some(f) => lookup.lookup(f.as_ref()).unwrap_or(lookup.unknown()),
        None => lookup.null(),
    }
}

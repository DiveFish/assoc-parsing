use std::fs::File;
use std::io::BufReader;

use failure::Error;
use rust2vec::{
    embeddings::Embeddings as R2VEmbeddings, io::ReadEmbeddings, storage::StorageWrap,
    vocab::VocabWrap,
};

use dpar::features::DependencyEmbeddings;

/// Read in focus and context embeddings on the basis of which the
/// association strength between two `AttachmentAddr`s will be calculated.
pub fn dep_embeds_from_files(
    focus_embeds: File,
    context_embeds: File,
) -> Result<(DependencyEmbeddings, DependencyEmbeddings), Error> {
    let mut focus_reader = BufReader::new(&focus_embeds);
    let focus_embeds: R2VEmbeddings<VocabWrap, StorageWrap> =
        R2VEmbeddings::read_embeddings(&mut focus_reader).unwrap();

    let mut context_reader = BufReader::new(&context_embeds);
    let context_embeds: R2VEmbeddings<VocabWrap, StorageWrap> =
        R2VEmbeddings::read_embeddings(&mut context_reader).unwrap();

    Ok((
        DependencyEmbeddings::from(focus_embeds),
        DependencyEmbeddings::from(context_embeds),
    ))
}

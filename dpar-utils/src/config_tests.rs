use std::fs::File;

use super::{Config, Lookup, Lookups, Model, Parser, TomlRead, Train};

lazy_static! {
    static ref BASIC_PARSER_CHECK: Config = Config {
        parser: Parser {
            pproj: true,
            system: String::from("stackproj"),
            inputs: String::from("parser.inputs"),
            transitions: String::from("parser.transitions"),
            no_lowercase_tags: vec!["TAG".to_string()],
            focus_embeds: String::from("parser.focus_embeds"),
            context_embeds: String::from("parser.context_embeds"),
            train_batch_size: 8000,
            parse_batch_size: 4000,
        },
        model: Model {
            graph: String::from("model.bin"),
            parameters: String::from("params"),
            intra_op_parallelism_threads: 4,
            inter_op_parallelism_threads: 6,
            allow_growth: true,
        },
        lookups: Lookups {
            word: Some(Lookup::Embedding {
                filename: String::from("word-vectors.bin"),
                normalize: true,
                op: String::from("word_op"),
                embed_op: String::from("word_embed_op"),
            }),
            tag: Some(Lookup::Embedding {
                filename: String::from("tag-vectors.bin"),
                normalize: true,
                op: String::from("tag_op"),
                embed_op: String::from("tag_embed_op"),
            }),
            deprel: Some(Lookup::Embedding {
                filename: String::from("deprel-vectors.bin.real"),
                normalize: false,
                op: String::from("deprel_op"),
                embed_op: String::from("deprel_embed_op"),
            }),

            chars: None,
            feature: None
        },
        train: Train {
            initial_lr: 0.05.into(),
            lr_scale: 0.5.into(),
            lr_patience: 4,
            patience: 15,
        }
    };
}

#[test]
fn test_parse_config() {
    let f = File::open("testdata/basic-parse.conf").unwrap();
    let config = Config::from_toml_read(f).unwrap();
    assert_eq!(*BASIC_PARSER_CHECK, config);
}

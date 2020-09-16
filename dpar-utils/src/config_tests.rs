use std::fs::File;

use super::{Config, Lookup, Lookups, Model, Parser, TomlRead, Train};

lazy_static! {
    static ref BASIC_PARSER_CHECK: Config = Config {
        parser: Parser {
            pproj: true,
            system: String::from("stackproj"),
            inputs: String::from("inputs/parser.inputs"),
            transitions: String::from("inputs/parser.transitions"),
            train_batch_size: 8192,
            parse_batch_size: 8192,
        },
        model: Model {
            graph: String::from("parser.graph"),
            parameters: String::from("parameters/epoch-0000"),
            intra_op_parallelism_threads: 2,
            inter_op_parallelism_threads: 2,
            allow_growth: true,
        },
        train: Train {
            initial_lr: 0.05.into(),
            lr_scale: 0.5.into(),
            lr_patience: 4,
            patience: 10,
        },
        lookups: Lookups {
            word: Some(Lookup::Embedding {
                filename: String::from("word-vectors.fifu"),
                op: String::from("model/tokens"),
                embed_op: String::from("model/token_embeds"),
            }),
            tag: Some(Lookup::Embedding {
                filename: String::from("tag-vectors.fifu"),
                op: String::from("model/tags"),
                embed_op: String::from("model/tag_embeds"),
            }),
            deprel: Some(Lookup::Table {
                filename: String::from("inputs/deprels.lookup"),
                op: String::from("model/deprels"),
            }),
            feature: Some(Lookup::Table {
                filename: String::from("inputs/features.lookup"),
                op: String::from("model/features"),
            }),
        }
    };
}

#[test]
fn test_parse_config() {
    let f = File::open("testdata/basic-parse.conf").unwrap();
    let config = Config::from_toml_read(f).unwrap();
    assert_eq!(*BASIC_PARSER_CHECK, config);
}

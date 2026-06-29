use bullet_lib::{
    game::{
        inputs::{ChessBucketsMirrored, get_num_buckets},
        outputs::MaterialCount,
    },
    nn::{
        InitSettings, Shape,
        optimiser::{AdamW, AdamWParams},
    },
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader::DirectSequentialDataLoader},
};

fn main() {
    let name = "mlt_01".to_string();
    // Arch
    let hl_size = 64;
    let l2_size = 16;
    #[rustfmt::skip]
    const BUCKET_LAYOUT: [usize; 32] = [
        0, 1, 2, 3,
        4, 4, 5, 5,
        6, 6, 6, 6,
        6, 6, 6, 6,
        7, 7, 7, 7, 
        7, 7, 7, 7, 
        7, 7, 7, 7, 
        7, 7, 7, 7
    ];
    const NUM_INPUT_BUCKETS: usize = get_num_buckets(&BUCKET_LAYOUT);
    const NUM_OUTPUT_BUCKETS: usize = 16;

    // Quantisation:
    const QA: i16 = 256;
    const QB: i16 = 128;
    const QC: i32 = 64;

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        .save_format(&[
            // merge in the factoriser weights
            SavedFormat::id("l0w")
                .transform(|store, weights| {
                    let factoriser = store.get("l0f").values.f32().repeat(NUM_INPUT_BUCKETS);
                    weights.into_iter().zip(factoriser).map(|(a, b)| a + b).collect()
                })
                .round()
                .quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            // this **is not** the format you want for fast inference,
            // but you can use `.transform` to transform it appropriately
            SavedFormat::id("l1w").transpose().round().quantise::<i8>(QB),
            SavedFormat::id("l1b").round().quantise::<i32>(QC),
            SavedFormat::id("l2w").transpose().round().quantise::<i32>(QC),
            SavedFormat::id("l2b").round().quantise::<i32>(QC*QC*QC), // 64^3 = 262144
            SavedFormat::id("l3w").transpose().round().quantise::<i32>(QC),
            SavedFormat::id("l3b").round().quantise::<i32>(QC*QC*QC*QC), // 64^4 = 16777216
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0f = builder.new_weights("l0f", Shape::new(hl_size, 768), InitSettings::Zeroed);
            let mut l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, hl_size);
            l0.init_with_effective_input_size(32);
            l0.weights = l0.weights + l0f.repeat(NUM_INPUT_BUCKETS);

            let l1 = builder.new_affine("l1", hl_size, NUM_OUTPUT_BUCKETS * l2_size);
            let l2 = builder.new_affine("l2", l2_size, NUM_OUTPUT_BUCKETS * 32);
            let l3 = builder.new_affine("l3", 32, NUM_OUTPUT_BUCKETS);

            // Faster version of
            // let stm_hidden = l0.forward(stm_inputs).crelu().pairwise_mul();
            // let ntm_hidden = l0.forward(ntm_inputs).crelu().pairwise_mul();
            let ft = |input, start, end| l0.slice(start, end).forward(input).crelu();
            let stm_hidden = ft(stm_inputs, 0, hl_size / 2) * ft(stm_inputs, hl_size / 2, hl_size);
            let ntm_hidden = ft(ntm_inputs, 0, hl_size / 2) * ft(ntm_inputs, hl_size / 2, hl_size);

            let hl1 = stm_hidden.concat(ntm_hidden);
            let hl2 = l1.forward(hl1).select(output_buckets).screlu();
            let hl3 = l2.forward(hl2).select(output_buckets).crelu();
            l3.forward(hl3).select(output_buckets)
        });

    // data
    let dataset_path = "data_shuffled.bin";

    // set optimiser params
    let optimiser_params = AdamWParams { beta1: 0.95, ..Default::default() };
    trainer.optimiser.set_params(optimiser_params);
    // stricter clipping so that values are still [-1.98, 1.98] despite factorizer
    let stricter_clipping = AdamWParams { beta1: 0.95, max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", stricter_clipping);
    trainer.optimiser.set_params_for_weight("l0f", stricter_clipping);
    trainer.optimiser.set_params_for_weight("l1w", stricter_clipping);

    // training configuration
    let superbatches = 50;
    let wdl_schedule = wdl::LinearWDL { start: 0.3, end: 0.6 };
    let lr_schedule = lr::CosineDecayLR {
        initial_lr: 0.001,
        final_lr: 0.001 * 0.3 * 0.3 * 0.3 * 0.3,
        final_superbatch: superbatches,
    };
    let lr_warmup = 200;

    let schedule = TrainingSchedule {
        net_id: name,
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        wdl_scheduler: wdl_schedule,
        lr_scheduler: lr::Warmup { 
            inner: lr_schedule,
            warmup_batches: lr_warmup 
        },
        save_rate: 10,
    };

    let settings = LocalSettings { threads: 12, test_set: None, output_directory: "checkpoints", batch_queue_size: 32 };

    let dataloader = DirectSequentialDataLoader::new(&[dataset_path]);

    //trainer.load_from_checkpoint("/mnt/d/bullet/checkpoints/cn_032-morelayer-480");
    trainer.run(&schedule, &settings, &dataloader);
    //trainer.save_quantised("checkpoints/cn_031-2step.nnue").expect("Failed to save quantised model");

    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ] {
        let eval = trainer.eval(fen);
        println!("FEN: {fen}");
        println!("EVAL: {}", 400.0 * eval);
    }
}

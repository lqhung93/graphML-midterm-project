from email.generator import Generator
import torch
import mmkgc
from mmkgc.config import Tester, AdvMixTrainer
from mmkgc.module.model import AdvMixRotatE
from mmkgc.module.loss import SigmoidLoss
from mmkgc.module.strategy import NegativeSampling
from mmkgc.data import TrainDataLoader, TestDataLoader
from mmkgc.adv.modules import MultiGenerator

from args import get_args

if __name__ == "__main__":
    args = get_args()
    print(args)
    # optional Weights & Biases logging (off unless -use_wandb 1)
    use_wandb = bool(args.use_wandb)
    if use_wandb:
        import wandb
        # bỏ tham số đường dẫn và tham số W&B khỏi config (đường dẫn lộ môi trường chạy)
        hidden = {"save", "resume", "resume_out"}
        wandb.init(
            entity=(args.wandb_entity or None),
            project=args.wandb_project,
            group=(args.wandb_group or None),
            notes=(args.wandb_notes or None),
            name=(args.wandb_name or ("%s_mu%s_seed%d" % (args.dataset, args.mu, args.seed))),
            config={k: v for k, v in vars(args).items() if k not in hidden and not k.startswith("wandb")},
            # không gửi metadata máy (tên máy, đường dẫn, requirements) lên W&B
            settings=wandb.Settings(x_disable_meta=True, x_disable_stats=True,
                                    x_disable_machine_info=True, x_save_requirements=False,
                                    disable_git=True, save_code=False, disable_code=True,
                                    disable_job_creation=True, host="gpu-node"),
        )
    # set the seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path="./benchmarks/" + args.dataset + '/',
        batch_size=args.batch_size,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=args.neg_num,
        neg_rel=0
    )
    # dataloader for test
    test_dataloader = TestDataLoader(
        "./benchmarks/" + args.dataset + '/', "link")
    img_emb = torch.load('./embeddings/' + args.dataset + '-visual.pth')
    text_emb = torch.load('./embeddings/' + args.dataset + '-textual.pth')
    # define the model
    kge_score = AdvMixRotatE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=args.dim,
        margin=args.margin,
        epsilon=2.0,
        img_emb=img_emb,
        text_emb=text_emb,
        ablation=(args.ablation or None)
    )
    print(kge_score)
    # define the loss function
    model = NegativeSampling(
        model=kge_score,
        loss=SigmoidLoss(adv_temperature=args.adv_temp),
        batch_size=train_dataloader.get_batch_size(),
    )
    
    adv_generator = MultiGenerator(
        noise_dim=64,
        structure_dim=2*args.dim,
        img_dim=2*args.dim
    )
    # train the model
    trainer = AdvMixTrainer(
        model=model,
        data_loader=train_dataloader,
        train_times=args.epoch,
        alpha=args.learning_rate,
        use_gpu=True,
        opt_method='Adam',
        generator=adv_generator,
        lrg=args.lrg,
        mu=args.mu,
        resume=args.resume,
        resume_out=args.resume_out,
        use_wandb=use_wandb
    )

    trainer.run()
    kge_score.save_checkpoint(args.save)

    # With chunked training the run is only finished once the full budget is spent;
    # evaluating a partial run wastes ~10 min on DB15K and reports a meaningless number.
    total = args.total_epoch or args.epoch
    if trainer.epochs_done < total:
        print("Session done: %d/%d epochs. Skipping evaluation until the budget is spent."
              % (trainer.epochs_done, total))
    else:
        # test the model
        kge_score.load_checkpoint(args.save)
        tester = Tester(model=kge_score, data_loader=test_dataloader, use_gpu=True)
        mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain=False)
        if use_wandb:
            # theo %, cùng đơn vị với các run W&B đã có của nhóm
            metrics = {"MRR": 100 * mrr, "MR": mr, "hit@10": 100 * hit10, "hit@3": 100 * hit3,
                       "hit@1": 100 * hit1}
            wandb.log(metrics)
            wandb.summary.update(metrics)
            wandb.finish()

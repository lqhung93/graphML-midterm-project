import argparse


def get_args():
    arg = argparse.ArgumentParser()
    arg.add_argument('-dataset', type=str, default='FB15K')
    arg.add_argument('-batch_size', type=int, default=1024)
    arg.add_argument('-margin', type=float, default=6.0)
    arg.add_argument('-dim', type=int, default=128)
    arg.add_argument('-epoch', type=int, default=1000)
    arg.add_argument('-save', type=str)
    arg.add_argument('-img_dim', type=int, default=4096)
    arg.add_argument('-neg_num', type=int, default=1)
    arg.add_argument('-learning_rate', type=float, default=0.001)
    arg.add_argument('-lrg', type=float, default=0.0001)
    arg.add_argument('-adv_temp', type=float, default=2.0)
    arg.add_argument('-visual', type=str, default='random')
    arg.add_argument('-seed', type=int, default=42)
    arg.add_argument('-missing_rate', type=float, default=0.8)
    arg.add_argument('-postfix', type=str, default='')
    arg.add_argument('-con_temp', type=float, default=0)
    arg.add_argument('-lamda', type=float, default=0)
    arg.add_argument('-mu', type=float, required=True)
    arg.add_argument('-adv_num', type=int, default=1)
    # chunked training: run -epoch epochs this session, out of -total_epoch overall.
    # -resume loads model+generator+both optimizers so a split run matches a continuous one.
    arg.add_argument('-total_epoch', type=int, default=None)
    arg.add_argument('-resume', type=str, default=None)
    arg.add_argument('-resume_out', type=str, default=None)
    # Ablation: bo 1 mo thuc khoi fusion ('no_s'|'no_v'|'no_t'); rong = day du
    arg.add_argument('-ablation', type=str, default='')
    # Weights & Biases (optional, off by default so existing runs are unchanged)
    arg.add_argument('-use_wandb', type=int, default=0)
    arg.add_argument('-wandb_project', type=str, default='gml-adamf-mat')
    arg.add_argument('-wandb_name', type=str, default='')
    arg.add_argument('-wandb_entity', type=str, default='')
    arg.add_argument('-wandb_group', type=str, default='')
    arg.add_argument('-wandb_notes', type=str, default='')
    return arg.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)

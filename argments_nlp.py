import argparse
import pprint



def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='rank')
    parser.add_argument('--dataset_type', type=str, default="MNLI")
    parser.add_argument('--data_path', type=str, default='./dataset/MNLI/')
    parser.add_argument('--qqp_data_path', type=str, default='./dataset/QQP/')
    parser.add_argument('--bert_path', type=str, default="bert-base-uncased")
    parser.add_argument('--encoder_type', type=str, default="cls")
    parser.add_argument('--max_length', type=int, default=128)#256
    parser.add_argument('--loss_weight_dis', type=float, default=0.001)
    parser.add_argument('--loss_weight_exp', type=float, default=0.0001)
    parser.add_argument('--rank_percent', type=float, default=0.05)
    parser.add_argument('--train_percent', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=5e-5)#2e-5
    parser.add_argument('--seed', type=int, default=550)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--log_name', type=str, default="1")
    parser.add_argument('--freeze_layer', type=int, default=0)#0
    parser.add_argument('--epoch_num', type=int, default=20)  #
    parser.add_argument('--batch_size', type=int, default=64)#
    parser.add_argument('--warmup_percent', type=float, default=0.1)  #0.1
    parser.add_argument('--rank', action="store_true")
    parser.add_argument('--gpu', type=int, default=6, help='which gpus to use')
    args = parser.parse_args()
    if args.rank:
        args.log_name = f'{args.name}_{args.rank_percent}'
    else:
        args.name = 'normal'
        args.log_name = args.name

    if args.dataset_type == "QQP":
        args.num_classes = 2
    else:
        args.num_classes = 3
    
    # pprint.PrettyPrinter().pprint(args.__dict__)
    return args

import argparse
import pprint


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='normal')
    parser.add_argument('--data_path', type=str, default='/share/liliz/imagenet/')
    
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--num_classes', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--epoch_num', type=int, default=200)  #
    parser.add_argument('--batch_size', type=int, default=128)  #
    parser.add_argument('--info_dim', type=int, default=1024)  #
    parser.add_argument('--milestones',
                        type=list,
                        default=[30, 60, 90, 120, 150], #[50, 100, 150], #0.0001 0.00001 0.000001 0.0000001
                        help='optimizer milestones') 
    
    parser.add_argument('--rank', action="store_true")
    parser.add_argument('--rank_percent', type=float, default=0.15)
    parser.add_argument('--warmup_percent', type=float, default=0.1) 

    parser.add_argument('--gpu', type=str, default='0', help='which gpus to use')
    args = parser.parse_args()
    if args.rank:
        args.log_name = f'{args.name}_{args.rank_percent}'
    else:
        args.name = 'normal'
        args.log_name = args.name
    
    # pprint.PrettyPrinter().pprint(args.__dict__)
    return args
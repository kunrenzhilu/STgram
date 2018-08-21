import argparse
import os

def get_parser(args):
    parser = argparse.ArgumentParser()
    
    #-- train
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--vocabulary_size', type=int, default=-1, help='please retrive the number of node accordingly')
    parser.add_argument('--num_negative_sample', type=int, default=64, help='number of negative sample used for sampled_softmax_loss, default 64')
    parser.add_argument('--num_epoch', type=int, default=40, help='the number of training epoch, should be used exclusively with the num_steps, default 50')
    parser.add_argument('--device', default='', type=str, help='which gpu device to use')
    parser.add_argument('--TIME_ONLY', default=False, action='store_true', help='using only time loss')
    parser.add_argument('--resume', default=False, action='store_true', help='resume experiment from checkpoint')
    parser.add_argument('--n_processes', type=int, default=3, help='Number of processes to be use, default 3')
    
    #-- model parameters
    parser.add_argument('--geo_dim', type=int, default=50, help='geographic embedding dimension, default 50')
    parser.add_argument('--sem_dim', type=int, default=50, help='semantic embedding dimension, default 50')
    parser.add_argument('--free_dim', type=int, default=0, help='the free dimension in the embedding')
    parser.add_argument('--normalize_weight', default=False, action='store_true', help='whether to normalize the embeddings during training')
    parser.add_argument('--geo_temp', type=float, default=10, help='the temperature for geo distance regularizor')
    parser.add_argument('--time_temp', type=float, default=0.01, help='the temperature for time difference regularizor')
    parser.add_argument('--geo_reg_type', type=str, choices=['l2', 'xn'], help='choose regularizor type between l2_loss or crossentropy loss')
    parser.add_argument('--regulation_weight', type=float, default=10, help='the regularization loss weight')
    parser.add_argument('--main_emb', type=str, default='emb', choices=['emb', 'weight', 'copy'], help='to experiment which embedding works better')
    
    #-- general
    parser.add_argument('--ROOT', type=str, default='/home/haibin2/data/checkins', help='root dir')    
    parser.add_argument('--LOG_DIR', type=str, help='experiment log directory')
    parser.add_argument('--CITY', type=str, default='TKY', choices=['NYC', 'TKY'], help='which city')
    parser.add_argument('--WITH_TIME', default=False, action='store_true',  help='whether the time is included')
    parser.add_argument('--WITH_USERID', default=False, action='store_true', help='whether to include the useid or not')
    parser.add_argument('--WITH_GPS', default=False, action='store_true', help='whether to include the gps or not')
    parser.add_argument('--WITH_TIMESTAMP', default=False, action='store_true', help='whether to convert the datetime object to timestamp')
    parser.add_argument('--pattern', type=str, default='hand', choices=['day-6', 'day-24', 'week-end', 'weekly', 'hand'], help='Used when --WITH_TIME, default day-24')
    
    args = parser.parse_args(args)
    
    if args.pattern == 'day-6':
        args.n_timeslot = 6
    elif args.pattern == 'day-24':
        args.n_timeslot = 24
    elif args.pattern == 'weekly':
        args.n_timeslot = 7
    elif args.pattern == 'week-end':
        args.n_timeslot = 2
    elif args.pattern == 'hand':
        args.n_timeslot = 6
    else: 
        args.n_timeslot = -1
    args.TB_DIR = os.path.join(args.LOG_DIR, 'tb')
    #args.EMB_DIR = os.path.join(args.LOG_DIR
    return args
    
    

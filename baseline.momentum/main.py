import os
from models import ThreeLayer
import torch
import argparse
from config import LogP, RBFKernelOnWeights
from SVGDConfig import SteinVariationalGradientDescentBase
from train import TrainOneTask, TrainOneTaskWithChaserLoss, test
from Dataset import Tasks
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import PolyLearningRatePolicy

parser = argparse.ArgumentParser()
parser.add_argument('-d', type=int, default=4, help='Which gpu to use')
parser.add_argument('-m', type=int, default=5, help='How many particles')
parser.add_argument('--weight_decay', default=1e-4, type = float, help='weight decay (default: 5e-4)')
parser.add_argument('--epoch', default=100000, type = int, help = 'The total number of iterations for meta-learning')
parser.add_argument('--step_size', default=1e-2, type = float, help = 'The step size for the inner fitting')
parser.add_argument('--steps', default=1, type = int, help = 'Number of iterations for the inner fitting')
parser.add_argument('--test_interval', default=200, type = int, help = 'How many iterations to between two test')
parser.add_argument('--nb_task', default=100, type = int, help = 'Number of K-shot tasks for training')
args = parser.parse_args()

# path ...
abs_path = os.path.realpath('./')
experiment_name = abs_path.split(os.path.sep)[-1]
log_dir = os.path.join('../../logs', experiment_name)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
    os.symlink(log_dir, './log')
#print(log_dir)
print('Experiement Name: {}'.format(experiment_name))
writer = SummaryWriter(log_dir)

DEVICE = torch.device('cuda:{}'.format(args.d))
print('Using Device: {}'.format(DEVICE))


# init all particles
net = ThreeLayer().to(DEVICE)
paramsvec0 = net.params2vec()

M = []
M.append(paramsvec0)
for i in range(args.m - 1):
    m = torch.nn.ParameterList([torch.nn.Parameter(torch.randn_like(pa) * 0.01) for pa in paramsvec0])
    M.append(m)

AllThetas = torch.nn.ParameterList()
for paramsvec in M:
    for param in paramsvec:
        AllThetas.append(param)
# init SVGD, kernel...
criterion = torch.nn.MSELoss().to(DEVICE)
logp = LogP(net, criterion)

#optimizer = torch.optim.SGD(paramsvec0, lr = 0.1, momentum = 0.9, weight_decay=args.weight_decay)
#optimizer = torch.optim.Adam(paramsvec0, lr = 0.02 * args.m, weight_decay=args.weight_decay) # !!!
optimizer = torch.optim.Adam(AllThetas, lr = 0.02 * args.m, weight_decay=args.weight_decay) # !!!
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30000, 50000, 70000, 90000], gamma=0.2)
GetInnerStepSize = PolyLearningRatePolicy(lr = args.step_size, max_iter = args.epoch, poly = 0.9)

kernel = RBFKernelOnWeights(1.0)
SVGD = SteinVariationalGradientDescentBase()
SVGD.Kernel = kernel
SVGD.NablaLogP = logp
SVGD.InitMomentumUpdaters(len(M))

tasks = Tasks(root = './data')

pbar = tqdm(range(args.epoch))
train_task_loader = tasks.TaskLoader(is_train = True, total_num_of_tasks = args.nb_task)
val_task_loader = tasks.TaskLoader(is_train = False, total_num_of_tasks = 100)
for i in pbar:
    lr_scheduler.step()

    args.step_size = GetInnerStepSize(i)

    train_task = next(train_task_loader)

    #ret_dic = TrainOneTask(task, M, SVGD, optimizer, DEVICE, args.steps, args.step_size)

    if i % args.test_interval == args.test_interval - 1:
        logp = test(val_task_loader, M, SVGD, DEVICE, args.steps, args.step_size)
        writer.add_scalar('TestLogP', logp, i // args.test_interval)
        print(logp)

    ret_dic = TrainOneTaskWithChaserLoss(train_task, M, SVGD, optimizer, DEVICE, args.steps, args.step_size)
    torch.cuda.empty_cache()
    pbar.set_description("Training")
    pbar.set_postfix(ret_dic)

    # For tensorboard
    LogpTrainDic = {}
    LogpJointDic = {}
    for key, var in ret_dic.items():
        if key.endswith('train'):
            LogpTrainDic.update({key:var})
    for key, var in ret_dic.items():
        if key.endswith('joint'):
            LogpJointDic.update({key:var})

    writer.add_scalars(main_tag='LogpTrain', tag_scalar_dict=LogpTrainDic, global_step=i)
    writer.add_scalars(main_tag='LogpJoint', tag_scalar_dict=LogpJointDic, global_step=i)
    # comment the next line if chaser loss is not used
    writer.add_scalar('ChaserLoss', ret_dic['chaser_loss'], global_step=i)





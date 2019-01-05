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
from utils import PolyLearningRatePolicy, AvgMeter

parser = argparse.ArgumentParser()
parser.add_argument('-d', type=int, default=7, help='Which gpu to use')
parser.add_argument('-m', type=int, default=5, help='How many particles')
parser.add_argument('--weight_decay', default=1e-4, type = float, help='weight decay (default: 5e-4)')
parser.add_argument('--iters', default=500, type = int, help = 'The total number of iterations for meta-learning')
parser.add_argument('--step_size', default=1e-2, type = float, help = 'The step size for the inner fitting')
parser.add_argument('--steps', default=5, type = int, help = 'Number of iterations for the inner fitting')
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

net = ThreeLayer().to(DEVICE)
paramsvec0 = net.params2vec()

M = []
M.append(paramsvec0)
for i in range(args.m - 1):
    m = torch.nn.ParameterList([torch.nn.Parameter(torch.randn_like(pa) * 0.01) for pa in paramsvec0])
    M.append(m)

criterion = torch.nn.MSELoss().to(DEVICE)
logp = LogP(net, criterion)


kernel = RBFKernelOnWeights(1.0)
SVGD = SteinVariationalGradientDescentBase()
SVGD.Kernel = kernel
SVGD.NablaLogP = logp
SVGD.InitMomentumUpdaters(len(M))

tasks = Tasks(root = './data')

train_task_loader = tasks.TaskLoader(is_train = True, total_num_of_tasks = args.nb_task)
val_task_loader = tasks.TaskLoader(is_train = False, total_num_of_tasks = 100)
raw_M = M
raw_lr = args.step_size
GetInnerStepSize = PolyLearningRatePolicy(lr=raw_lr, max_iter=args.iters, poly=0.9)

logps = AvgMeter()

for iii in range(10):
    pbar = tqdm(range(args.iters))
    task = next(val_task_loader)
    X, Y, Xtest, Ytest, std = task
    X = X.to(DEVICE)
    Y = Y.to(DEVICE)
    Xtest = Xtest.to(DEVICE)
    Ytest = Ytest.to(DEVICE)
    std = std.to(DEVICE) * 100  # * 100 to stablize
    SVGD.NablaLogP.update(X, Y, std)
    SVGD.InitMomentumUpdaters()
    for i in pbar:

        args.step_size = GetInnerStepSize(i)

        #print('i, len', i, len(M))
        M = SVGD.step(M, retain_graph=False, step_size=args.step_size)

        for paramsvec in M:
            for param in paramsvec:
                param.detach_()
                param.requires_grad = True
        with torch.no_grad():
            logp = 0
            for paramsvec in M:
                logp = logp + SVGD.NablaLogP(True, paramsvec, ret_grad=False)

        logp.detach_()
        logp = logp.item() / len(M)

        writer.add_scalar('SVGDLogP', logp, i + iii*args.iters)
        #print(logp)
        #torch.cuda.empty_cache()
        pbar.set_description("SVGD fitting")
        pbar.set_postfix({'inner_lr':args.step_size, 'logp':logp})
    torch.cuda.empty_cache()
    logps.update(logp)
    M = raw_M

print(logps.mean)

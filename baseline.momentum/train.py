import torch
from collections import OrderedDict
from tqdm import tqdm
from utils import AvgMeter

def TrainOneTask(Task, M, SVGD, optimizer, DEVICE, num_of_step = 3, step_size = 1e-3):
    X, Y, Xtest, Ytest, std = Task
    X = X.to(DEVICE)
    Y = Y.to(DEVICE)
    Xtest = Xtest.to(DEVICE)
    Ytest = Ytest.to(DEVICE)
    std = std.to(DEVICE) * 100  # * 100 to stablize

    SVGD.NablaLogP.update(X, Y, std)
    SVGD.InitMomentumUpdaters()

    with torch.no_grad():
        start_logp = 0
        for paramsvec in M:
            start_logp = start_logp + SVGD.NablaLogP(True, paramsvec, ret_grad=False)
    start_logp = start_logp / len(M)
    for i in range(num_of_step):
        M = SVGD.step(M, retain_graph = True, step_size = step_size)

    with torch.no_grad():
        end_logp = 0
        for paramsvec in M:
            end_logp = end_logp + SVGD.NablaLogP(True, paramsvec, ret_grad = False)
    end_logp = end_logp / len(M)
    SVGD.NablaLogP.update(Xtest, Ytest, std)
    logp = 0
    for paramsvec in M:
        logp = logp + SVGD.NablaLogP(True, paramsvec, ret_grad=False)
    logp = logp / len(M)
    loss = logp * -1.0
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    ret_dic = OrderedDict()
    ret_dic['start_logp_train'] = start_logp.item()
    ret_dic['end_logp_train'] = end_logp.item()
    ret_dic['end_logp_joint'] = logp.item()

    return ret_dic


def TrainOneTaskWithChaserLoss(Task, M, SVGD, optimizer, DEVICE, num_of_step = 3, step_size = 1e-3):
    optimizer.zero_grad()
    X, Y, Xtest, Ytest, std = Task
    X = X.to(DEVICE)
    Y = Y.to(DEVICE)
    Xtest = Xtest.to(DEVICE)
    Ytest = Ytest.to(DEVICE)
    std = std.to(DEVICE) * 100  # * 100 to stablize

    SVGD.NablaLogP.update(X, Y, std)
    SVGD.InitMomentumUpdaters()
    # Compute the LogP for initial particles  (For hyper-param tuning)
    with torch.no_grad():
        start_logp = 0
        for paramsvec in M:
            start_logp = start_logp + SVGD.NablaLogP(True, paramsvec, ret_grad=False)
    start_logp = start_logp / len(M)
    # Inner fit
    for i in range(num_of_step):
        M = SVGD.step(M, retain_graph = True, step_size = step_size)

    # Compute the LogP of the training set after the fitting (For hyper-param tuning)
    with torch.no_grad():
        end_logp = 0
        for paramsvec in M:
            end_logp = end_logp + SVGD.NablaLogP(True, paramsvec, ret_grad = False)
    end_logp = end_logp / len(M)
    Xtrain_and_test = torch.cat((X, Xtest))
    Ytrain_and_test = torch.cat((Y, Ytest))
    SVGD.NablaLogP.update(Xtrain_and_test, Ytrain_and_test, std)
    SVGD.InitMomentumUpdaters()

    # Compute the LogP of the whole set after the fitting (For hyper-param tuning)
    with torch.no_grad():
        logp = 0
        for paramsvec in M:
            logp = logp + SVGD.NablaLogP(True, paramsvec, ret_grad=False)
    logp = logp / len(M)
    # Approximate the true prior

    M_true = []
    for paramsvec in M:
        m = torch.nn.ParameterList([torch.nn.Parameter(p.detach()) for p in paramsvec])
        #m = [p.detach() for p in paramsvec]
        M_true.append(m)

    #M_true = SVGD.step(M, retain_graph=False, step_size=step_size)
    for i in range(num_of_step):
        M_true= SVGD.step(M_true, retain_graph=False, step_size=step_size)

    chaser_loss = 0
    for paramsvec, paramsvec_true in zip(M, M_true):
        for param, param_true in zip(paramsvec, paramsvec_true):
            chaser_loss = chaser_loss + torch.mean((param - param_true.detach()) ** 2)

    chaser_loss = chaser_loss / len(M)
    # Compute the true LogP of the whole set (For hyper-param tuning)
    with torch.no_grad():
        true_logp = 0
        for paramsvec in M_true:
            true_logp = true_logp + SVGD.NablaLogP(True, paramsvec, ret_grad=False)
    true_logp = true_logp / len(M)
    chaser_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    ret_dic = OrderedDict()
    ret_dic['start_logp_train'] = start_logp.item()
    ret_dic['end_logp_train'] = end_logp.item()
    ret_dic['end_logp_joint'] = logp.item()
    ret_dic['true_logp_joint'] = true_logp.item()
    ret_dic['chaser_loss'] = chaser_loss.item()

    return ret_dic


def test(TaskLoader, M, SVGD, DEVICE, num_of_step = 3, step_size = 1e-3):

    raw_M = M
    LogP = AvgMeter()
    pbar = tqdm(range(100))
    for i in pbar:
        task = next(TaskLoader)
        X, Y, Xtest, Ytest, std = task
        X = X.to(DEVICE)
        Y = Y.to(DEVICE)
        Xtest = Xtest.to(DEVICE)
        Ytest = Ytest.to(DEVICE)
        std = std.to(DEVICE) * 100  # * 100 to stablize

        SVGD.NablaLogP.update(X, Y, std)
        SVGD.InitMomentumUpdaters()

        #Mt = SVGD.step(M, retain_graph=False, step_size=step_size)
        for i in range(num_of_step):
            M = SVGD.step(M, retain_graph = False, step_size = step_size)

        SVGD.NablaLogP.update(Xtest, Ytest, std)
        SVGD.InitMomentumUpdaters()

        with torch.no_grad():
            logp = 0
            for paramsvec in M:
                logp = logp + SVGD.NablaLogP(True, paramsvec, ret_grad=False)
        logp = logp / len(M)
        LogP.update(logp.item())
        pbar.set_description("Running Validation")
        pbar.set_postfix({'Logp_test':LogP.mean})

        M = raw_M
    return LogP.mean

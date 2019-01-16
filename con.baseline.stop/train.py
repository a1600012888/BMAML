import torch
from collections import OrderedDict
from tqdm import tqdm
from utils import AvgMeter

def TrainFewTaskOneStep(Tasks, M, SVGD, optimizer, DEVICE, num_of_step = 3, step_size = 1e-3):
    optimizer.zero_grad()

    raw_M = M
    HistroyThetas = [[] for i in range(num_of_step)] # used for recurrent kernel
    # HistoryThetas[i][j] = particles in the i-th iteration of SVGD fitting for task j
    # HistoryThetas - [ [particles after step1], [particles after step2] , [particles after step3]]
    # [params after step1] = [fit for task1, fit for task2, fit for task3, fit for task4..]

    mean_loss = 0
    for i in range(len(Tasks) - 1):
        #print(i)
        now_task = Tasks[i]
        next_task = Tasks[i+1]
        X, Y, Xtest, Ytest, std = now_task
        X = X.to(DEVICE)
        Y = Y.to(DEVICE)
        std = std.to(DEVICE) * 100  # * 100 to stablize

        nextX, nextY, nextXtest, nextYtest, nextstd = next_task
        nextXtest = nextXtest.to(DEVICE)
        nextYtest = nextYtest.to(DEVICE)
        nextstd = nextstd.to(DEVICE) * 100  # * 100 to stablize

        SVGD.NablaLogP.update(X, Y, std)
        SVGD.InitMomentumUpdaters()

        for j in range(num_of_step):

            if len(HistroyThetas[j]) > 0:
                for paramsvec in HistroyThetas[j][-1]:
                    for param in paramsvec:
                        param.detach_()
            HistroyThetas[j].append(M)
            M = SVGD.step(HistroyThetas[j], retain_graph = True, step_size = step_size)
            HistroyThetas[j][-1] = M
            #HistroyThetas[j].append(M)

        SVGD.NablaLogP.update(nextXtest, nextYtest, nextstd)

        logp = 0
        for paramsvec in M:
            logp = logp + SVGD.NablaLogP(True, paramsvec, ret_grad=False)
        logp = logp / len(M)
        loss = logp * -1.0

        loss.backward()
        mean_loss = mean_loss + loss.item()

        M = raw_M

    optimizer.step()
    optimizer.zero_grad()
    ret_dic = OrderedDict()

    ret_dic['mean_loss'] = mean_loss / (len(Tasks) -1)

    return ret_dic

def TrainFewTaskFewStep(Tasks, M, SVGD, optimizer, DEVICE, num_of_step = 3, step_size = 1e-3):
    optimizer.zero_grad()

    HistroyThetas = [[] for i in range(num_of_step)] # used for recurrent kernel
    # HistoryThetas[i][j] = thetas in the i-th iteration of SVGD fitting for task j
    # for the i-th step of SVGD, task1,task2,task3, taskj
    # HistoryThetas - [ [params after step1], [params after step2] , [params after step3]]
    # [params after step1] = [fit for task1, fit for task2, fit for task3, fit for task4..]

    ret_dic = OrderedDict()
    for i in range(len(Tasks) - 1):
        #print(i)
        now_task = Tasks[i]
        next_task = Tasks[i+1]
        X, Y, Xtest, Ytest, std = now_task
        X = X.to(DEVICE)
        Y = Y.to(DEVICE)
        std = std.to(DEVICE) * 100  # * 100 to stablize

        nextX, nextY, nextXtest, nextYtest, nextstd = next_task
        nextXtest = nextXtest.to(DEVICE)
        nextYtest = nextYtest.to(DEVICE)
        nextstd = nextstd.to(DEVICE) * 100  # * 100 to stablize

        SVGD.NablaLogP.update(X, Y, std)
        SVGD.InitMomentumUpdaters()

        for j in range(num_of_step):

            # If stop gradient
            if len(HistroyThetas[j]) > 0:
               for paramsvec in HistroyThetas[j][-1]:
                    for param in paramsvec:
                        param = param.detach() # no detach_

            HistroyThetas[j].append(M)
            M = SVGD.step(HistroyThetas[j], retain_graph = True, step_size = step_size)
            HistroyThetas[j][-1] = M  # out put..

        SVGD.NablaLogP.update(nextXtest, nextYtest, nextstd)


        if i == (len(Tasks) - 2):
            logp = 0
            for paramsvec in M:
                logp = logp + SVGD.NablaLogP(True, paramsvec, ret_grad=False)
            logp = logp / len(M)
            loss = logp * -1.0
            loss.backward()
            train_loss = loss.item()
        else:
            with torch.no_grad():
                logp = 0
                for paramsvec in M:
                    logp = logp + SVGD.NablaLogP(False, paramsvec, ret_grad=False)
                logp = logp / len(M)
                ret_dic['task_{}_logp'.format(i)] = logp.item()
        #else:
        #    loss.backward(create_graph = True)
        #mean_loss = mean_loss + loss.item()

        #M = HistroyThetas[0][0]

    optimizer.step()
    optimizer.zero_grad()

    ret_dic['train_logp'] = train_loss * -1#/ (len(Tasks) -1)

    return ret_dic


def test_con(TaskLoader, M, SVGD, DEVICE, num_of_step = 3, step_size = 1e-3):
    raw_M = M

    LogP = AvgMeter()
    pbar = tqdm(range(100))
    for t in pbar:
        pbar.set_description("Validation")
        M = raw_M
        Tasks = next(TaskLoader)
        HistroyThetas = [[] for i in range(num_of_step)] # used for recurrent kernel
        for i in range(len(Tasks) - 1):
            now_task = Tasks[i]
            next_task = Tasks[i+1]
            X, Y, Xtest, Ytest, std = now_task
            X = X.to(DEVICE)
            Y = Y.to(DEVICE)
            std = std.to(DEVICE) * 100  # * 100 to stablize

            nextX, nextY, nextXtest, nextYtest, nextstd = next_task
            nextXtest = nextXtest.to(DEVICE)
            nextYtest = nextYtest.to(DEVICE)
            nextstd = nextstd.to(DEVICE) * 100  # * 100 to stablize

            SVGD.NablaLogP.update(X, Y, std)
            SVGD.InitMomentumUpdaters()

            for j in range(num_of_step):

                HistroyThetas[j].append(M)
                M = SVGD.step(HistroyThetas[j], retain_graph = False, step_size = step_size)
                HistroyThetas[j][-1] = M
                #HistroyThetas[j].append(M)

            SVGD.NablaLogP.update(nextXtest, nextYtest, nextstd)

            if i == (len(Tasks) - 2):
                logp = 0
                for paramsvec in M:
                    logp = logp + SVGD.NablaLogP(True, paramsvec, ret_grad=False)
                logp = logp / len(M)
                LogP.update(logp.item())
        pbar.set_postfix({'logp': LogP.mean})
    return LogP.mean


import os

import sys

import string
import tensorboardX
import numpy as np
import matplotlib.pyplot as plt
if __name__ == "__main__":

    n = 1

    cmd = ""

    if len(sys.argv) >= 2:
        n = int(sys.argv[1])
        #cmd = ' '.join(sys.argv[2:])
    else:
        print("error")
        exit(1)
    cmds = ["python train.py task=FactoryTaskNutBoltPick checkpoint=./runs/FactoryTaskNutBoltPickckpt_sequence_2.pt.pth test=True",
            "python train.py task=FactoryTaskNutBoltPlace checkpoint=./runs/FactoryTaskNutBoltPlaceckpt_sequence_2.pt.pth test=True",
            "python train.py task=FactoryTaskNutBoltScrew checkpoint=./runs/FactoryTaskNutBoltScrew/nn/last_FactoryTaskNutBoltScrew_ep_7400_rew_-0.786197.pth test=True"]
    w1 = tensorboardX.SummaryWriter("./runs/FactoryTaskNutBoltPick_test_random/nut_bolt_size")
    for index, cmd in enumerate(cmds):
        task = cmd.split("task=")[1]
        task = task.split(" ")[0]
        #t1 =tensorboardX.SummaryWriter("./runs/"+task+"_test_random/density")
        de_s = []
        sr_list = []
        plt.figure(index)
        de2sr = {}
        for x in range(n):
            out = os.popen(cmd)
            out = out.readlines()
            s_r = 0
            for o in out:
                if "successe rate : " in o:
                    s_r = float(o.split("successe rate : ")[1].split("\n")[0])
                    sr_list.append(s_r)

                    #t1.add_scalar("density_randomization", s_r, de)
        #plt.scatter(np.array(de_s),np.array(sr_list),c="blue", label="overall_sr:%.2f"%np.mean(sr_list))
        #plt.legend(loc='best')
        #plt.savefig(task)

        print(task+"_nut_bolt_m8_tight : %f"%sr_list[0])
    #t1.add_scalar("all_success_rate",np.mean(sr_list), 1)
    #t1.close()
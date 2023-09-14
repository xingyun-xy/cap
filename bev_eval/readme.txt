1. 保存模型的预测结果
   Bevdepth工程里base_exp.py中eval_step()函数输出预测结果,将预测结果直接保存成npy文件		
   np.save("tmp.npy",test_step_outputs)
	
2. python test_3D.py 输入npy文件和结果保存地址
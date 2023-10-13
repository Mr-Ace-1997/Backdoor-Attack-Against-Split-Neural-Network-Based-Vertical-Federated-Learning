# Backdoor-Attack-Against-Split-Neural-Network-Based-Vertical-Federated-Learning

This code is the attack scheme in the paper "Backdoor Attack Against Split Neural Network-Based Vertical Federated Learning". We supply this version as a reference of our attack scheme. 

Among the files, the model structure and data processing is applicable for CIFAR10. You can download the CIFAR10 data by yourself and run our codes on it directly.

You can test the attack scheme as following steps:

1. Generate a clean model to validate the baseline classification accuracy:

   > python clean_train.py --clean-epoch 80 --dup 0 --multies 4 --unit 0.25
   
   You can run `python clean_train.py -h` to view the meaning of each arguments.
   
   After this step, you will obtain a clean model which completes a full training of 100 epochs, along with a pre-poisoned model which completes clean 80 epochs for the backdoor attack.
   
2. Poison the model and generate the special trigger vector:

   > python poison_train.py --label 0 --dup 0 --magnification 6 --multies 4 --unit 0.25 --clean-epoch 80

   In this step, the process will poison and train on the pre-poisoned model in the step 1 for the remained 20 epochs. Please note that the argument values of `dup`, `multies`, `unit` and `clean-epoch` need to be consistent with those in the step 1.

   After this step, you will obtain a backdoored model which completes a full training of 100 epochs, along with its trigger vector.

4. Add appropriate noise on the trigger vector to avoid its repeat appearences in the uploading process of the bottom model:

   > python add_noise.py --multies 4 --unit 0.25

   You can see the attack successful rate of the trigger vector with noise.

   Note that you can open the `noise_vec_0.csv` and `normal_vec_0.csv` to directly observe the small differences between them and adjust the size of the noise to the appropriate range based on our paper.

Feel free to contact me (guapi7878@gmail.com) if you have any questions.

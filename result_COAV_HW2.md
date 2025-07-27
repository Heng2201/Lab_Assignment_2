# 08/29 Results of COAV_HW2

First, I trained two different models because the model I used in the beginning can't present a nice results on testing accuracy. The structure of this two models will be shown at below.

<details>
<summary>Model 1 - The initial model: 3 convolution layers and 3 fully connected layers </summary>

(image of structure one)

</details>
<br>
<details>
<summary>Model 2 - Model only use convolution layers</summary> 
This structure is from <a href="https://arxiv.org/pdf/1412.6806v3.pdf">this article</a>. In the article they say the structure I copied can obtain accuracy around 90% of accuracy, but the fact is I only get around 80%. 

(image of structure 2)

</details>
<br>

> ### Both model contains 
- momentum = 0.9 
- learning scheduler that decrease learning rate by 90% every 30 epochs
- input dropout with p=0.2 
- dropout layer with p=0.5 after each pooling  
- training data augmentation using torchvision built-in CIFAR10 `AutoAugmentPolicy` 

Because the second model always has a better result on testing accuracy compare to the first model (around 3 to 4%). So in this issue I will only show results from the second model.

## result for different batch size
Below shows the result of accuracy and training loss when batch size are 4 and 10. 

<details>
<summary>Result on accuracy and loss when different batch size</summary>

- batch size=4
<img src=".\useful pic\b4_model2.png" width = '600' >

- batch size=10
<img src=".\useful pic\b10_model2.png" width = '600'>

</details>
<br>


## result for different learning rate
Below shows the result of accuracy and training loss when learning rate are 0.001 and 0.0005 with batch size=10. 


<details>
<summary>Result on accuracy and loss with different learning rate</summary>

- learning rate = 0.001
<img src=".\useful pic\b10_model2.png" width = '600'>

- learning rate = 0.0005
<img src=".\useful pic\model2_b10_lr0.0005.png" width='600'>

</details>
<br>

## Result of random 10 images prediction
 Figure below shows the results of the prediction on 10 random images by the model. In the figure, `p` means prediction done by the model and `a` means actual class of the images.

<details>
<summary>Result</summary>
<img src=".\useful pic\random_10.png" width='600'>

</details>

In addition to the overall accuracy, we should also look at the classes in which the model performs weakly and in which it performs strongly. Therefore, I also calculated the model's accuracy for each class and the results are shown below.

<details>
<summary>Accuracy on each class</summary>
<img src=".\useful pic\set_accuracy.png" width='600'>

</details>

Can see that our model is weak at predicting cats and dogs, I think this is because cat and dog look very similar to the model.
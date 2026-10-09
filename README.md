# 26trustai_jlens-overfitters
## Meetings
### 09.10.2026
Multihop (need of multiple step to find the answer) reasoning dataset

How can we quantify this, you want to quantify it? graph? 

Maybe in later time, find temporal patterns in the multihop, ...

New paper Suspicious, applying some filters to the data that are learn. His guess, it is not really generalistic. 

Held-out / Test scores, so unseen tokens / out-of-distribution prompts.

For next meeting:
Reproduce
Analysis on the multihop things


Gert notes:
Given z = 5, and x = z + 3, what is x?[J-Lens Readout]
 
JLens:
[
8
token2
5
token3
...
]
 
Given z = 5, and x = z + 3, what is x?
First do some reasoning:
I substitute z and get x = 8
[J-Lens Readout]
 
JLens:
[
8 
token2
token3
token4
5
]
 
 
MultiHop reasoning dataset 
[Text about Louis]
What is the birthdate of the father of Louis XIV
Louis -> Father -> Birthdate 
CoT: The father of Louis is Louis, his birthdate by the text given is 1793.
 
 
J++ Lens
Its improvements over the J-Lens, they apply 2 tricks to make it "better" 
However, they also define some filtering function, which is learned.
 
Held-out / Test scores, so unseen tokens / Out-of-Distribution prompts.


Next meeting is the Tuesday 20th of October at 2PM.
Task are: Finish reading main paper, read new paper, reproduce the paper and need to recall all my memory but something about the multihop reasoning

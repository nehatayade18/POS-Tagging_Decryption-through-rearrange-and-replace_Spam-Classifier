
## Part 1  POS Tagging
Problem Description:-
	Part of speech tagging is the process of parsing through a sentence and assigning correct part of speech to each word of the sentence. For this we are provided with a dataset to train our model and then make predictions on the test sentences.

Approach:-
	For performing POS Tagging we have implemented three techiniques. 
	1) Using Simple Bayes Net 
	2) Using Viterbi Algorithm 
	3) Using Gibbs Sampling
	
	For implementing this three techiniques we have used prior probability, emission probability, posterior 
	probability, transition probability and initial probability.
	Prior Probability:  P(Si) = Probability of each tag.
						P(Si) = Count(Si)/Count(Total S)
	Emission Probability:	P(Wi|Si) = Probability of word given tag.
							P(Wi|Si) = Count(Wi,Si)/Count(Total Si)
	Posterior Probability:  P(Si|Wi) = Probability of tag given word.
							P(Si|Wi) = P(Wi|Si)*P(Si)
	Transition Probability: P(Si+1|Si) = Probability of next tag given current tag.
							P(Si+1|Si) = Count(Si,Si+1)/Count(Total Si)
	Initial Probability : P(Wi) = Probability of a sentence starting from word W.
						  P(Wi) = Count(Wi such that it is first word)/Count(Total Wi)

	Here, S = Tags
		  W = Words
		  Posterior probability is used in simple bayes net model
		  Initial probability, transition probability and emission probability are used in viterbi model
		  Posterior probability, emission probability and transition probability are used in gibbs model

	1) Simple Bayes Net: Here, we are calculating posterior probability for each word in the sentence and 
	selecting the one with max probability. 
	For log posterior we return the sum of log of all the probabilities of the predicted tags.

	2) Viterbi Algorithm: Here, we are calculating the probability for each state, that is the word, in a 
	forward pass and then selecting the most probable tag in the backward pass. 
	For this we take product of initial probability and emission probability for the W0 and then for W1 to 
	Wi we take product of previous probability and transition probability. We store this probabilities in a 
	list and in the backward pass select the one with max probabilities. 
	For log posterior we return the of log of max probabilities of the predicted tags.

	3) Gibbs Sampling: Here, first we are initializing the tags for each word in sentence as 'noun', it can be 
	initialized to any random tags but we are using noun since it is the most occuring tag in the dataset. Then 
	we calculate probability for each tag and use a random value to choose a tag from this list of probabilities 
	and update this tag with the initially assigned 'noun'. We repeat this process for 1000 iterations and then 
	return the final updated list of predicted tags containing most frequent tag for each word. 
	Now to calculate the probabilities, we take product of emission probability and prior for the W0, for W1 to Wi-1 
	we product of previous probability and transition probability, and for Wi we take product of previous probability, 
	transition probability and transition probability P(S0|Si). For log posterior we keep track of all the probabilities 
	of the sample and return the sum of probabilities of the tags that are returned in the updated list of predicted tags.

Results:-
	
	Attached with files.
	For bc.test = Final1.png
	For bc.test.tiny = Final2.png
	
Observations:-

	After implementing above mentioned algorithms while fine tuning the models we observed that we can imporve the 
	accuracy by handling the words that are not in the dataset but are in the test sentence. The approach we used 
	is, since "noun" is most frequent tag in the dataset we modified the emission probabilty function in way that if 
	we encounter a word that is not in the dataset while calculating the probabilities tag "noun" will be given some 
	extra weight. Even though this is extremely random approach we saw that there was considerable increase in the 
	accuracy. But what it also did was unlike our previous results where Best was viterbi than gibbs and at last simple 
	bayes net, the simple bayes net model started making pretty good predictions and got accuracy almost similar to that 
	of gibbs. This results are shown below:
	Fine tuned results = Final3.png
	
## Part 2 Decryption through replace and rearrangement of alphabets
Probability Matrix:
Likelihood: It takes corpus as input to create a square matrix of size 26 by 26 initialized with zeroes for total set of characters. If the corpus is blank for previous character and next character, transition value is 0. For current value if space but previous is not space, transition value s ASCII value of previous. Similarly, if current character in corpus is not space but previous value is a space, transition value is ord of previous character. Last case when both the characters are non space value, transition value is the ord of difference of both ASCII values.  
Prior: It is the probability of each character(P(W0(i) )) for which transitions are calculated. For each row, transition sum of the values from that character upon its occurrences in the whole matrix is calculated.
Score Calculation: Curr is a temporary variable which takes the value of characters in the encoded document. So if the current value in the encrypted text is a space, then values for rows remain unchanged or else for any other character(P(Wi) P(Wi+1)), the value is the difference in the ASCII values of the current and transition character(ord(prev character of encrypted)-ord('a')). 
For the following four cases of (P(Wi) P(Wi+1)
1.	If previous encoded and encoded alphabet is space , score remains unchanged
2.	If previous character is blank and encoded alphabet is not space , score is column index difference of Ascii values of current and ‘a’ alphabet(removing initial gap which starts from 97)
3.	If encoded is not blank, transition score is ASCII difference of encoded character and ‘a’
4.	If current and previous character is not space, transition value is (encoded character- previous character) ASCII value.

Total score :   score = ∑ ( P (w(i), P(w(i+1)) )

Replace table: It randomly shuffles the lower case alphabets in the list to replace the pass characters of range 27 with some random character. Here the values of characters get swapped. A table with swapped character values is returned by the function

Rearrange table: It considers a window of 4 numbers as indices for the selected set of characters and randomly samples to swap indices of 2 characters as done with the encryption table. Several combinations of such swapped indices are checked during multiple iterations to try out different permutations of the selected text.

Break code: It takes the encrypted file and corpus as input and then generates a probability matrix for corpus alphabet to alphabet transition. Then the replace and rearrange tables are initialized to give out the initial orientation of the document and the likelihood score of the document.  Out of the two methods of replace and rearrange, one of them is selected in the for loop. For the replaced table, rearrange able is untouched and vice-versa. The old doc is decrypted once with the initialized tables and the new document is calculated from the replaced/rearranged tables. New/proposed document Is decrypted using the replaced and rearranged tables. Likelihood of both the documents is calculated from the documents passed and the probability matrix. Tables are replaced with the newly shuffle tables and the best document matrix with greater score (ie,if P(D_new)>P(D_old)) is stored in the matrix as tuple of new document probability and the document itself. 
Best document[0] = Probability of new document with greater score
Best document[1] = New document
For each document, the P(D) =∑ P(Wi) , is calculated. A final document is created which is a list of all the best documents obtained so far. Through the matrix, final document with greater probability score is selected. The final stream of decrypted characters in the output file inserted is the document with the highest probability score . 



## Part 3 Mail Spam Classifier

To solve this spam filtering problem we use a Naive Bayes classifier to infer 
whether an email is spam or not using the distribution of words in the new message. 

The classifier makes the Naive Bayes assumption, where it models each word in the message 
as being conditionally independent of the rest of the words given a Spam/NotSpam variable.
This allows us to compute the likelihood of a word given spam by a product of likelihoods 
of the individual words: `P(w1|Spam) * P(w2|Spam) * etc...`, which is a convenient factorization 
of the joint probably. 

We estimate these likelihoods by building a table of word-given-spam probabilities from training 
data. We also keep a table of word-given-notspam probabilities. At test time we compute the ratio 
of likelihoods between spam and notspam 

We include a number of optimizations to this implementation:

 - Filtering stop words (e.g. `"the"`, `"and"`, `"of"`, etc...)
 - Computing log probabilities to prevent numerical underflow
 - Computing full word frequency distributions, rather than just number of spam documents with a given word
 
On the provided test set we get 96% classification accuracy. 

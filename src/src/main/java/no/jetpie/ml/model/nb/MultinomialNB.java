package no.jetpie.ml.model.nb;

import no.uib.cipr.matrix.*;
import no.jetpie.ml.feature.Vectorizer;
/**
 * The multinomial Naive Bayes model
 * 
 * @author bingqingqu
 * @version 0.1.0
 * @date 2015.1.13
 *
 */
public class MultinomialNB extends NaiveBayes{
	
	public MultinomialNB(Vectorizer v,String filepath, String boundary){
		super(v,filepath,boundary);
	}
	
	public MultinomialNB(String vocabpath,String filepath, String boundary){
		super(vocabpath,filepath,boundary);
	}	
	
	@Override
	protected Matrix jointLogLikelihood(Matrix X) {

		Matrix jil= new DenseMatrix(X.numRows(),this.numCats);
		
		// get log(likelihood)
		X.mult( this.TfeatureCondProb,jil);		
		
		// get log(prior) + log(likelihood)
		for(MatrixEntry e: jil)
			e.set(e.get() + this.logPrior);
		return jil;
	}

}

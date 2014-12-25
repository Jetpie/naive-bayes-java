package src.main.core.nb;

import java.util.*;
import java.util.Vector;
import java.util.concurrent.TimeUnit;


import no.uib.cipr.matrix.*;
import no.uib.cipr.matrix.sparse.*;

import src.main.core.feature.Vectorizer;
/**
 * The multinomial Naive Bayes model
 * 
 * @author bingqingqu
 * @version 0.1.0
 * @date 2014.12.08
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
		// reset the timetiker
		this.stopwatch.reset();
		this.stopwatch.start();
		
		Matrix jil= new DenseMatrix(X.numRows(),this.numCats);
		
		// get log(likelihood)
		X.mult( this.TfeatureCondProb,jil);		
		
		// get log(prior) + log(likelihood)
		for(MatrixEntry e: jil)
			e.set(e.get() + this.logPrior);
		long millis = stopwatch.elapsed(TimeUnit.MILLISECONDS);
		System.out.println("(Elasped time for prediction: " + millis + " ms)");
		return jil;
	}	
	
	
	public static void main(String [] args){
		
		
//		Vectorizer v = new Vectorizer("vocabulary.model",true);
//		NaiveBayes nb = new MultinomialNB(v,"proba/");
		String boundary =  "boundary.json";
		String model = "log_proba/";
		String vocabulary = "vocabulary.model";
		NaiveBayes nb = new MultinomialNB(vocabulary,model,boundary);
		String sen = "贝 珍珠 恒美 绽放 耳钉 - 黑色   首页  >  Mbox 饰品 专场  >  贝 珍珠 恒美 绽放 耳钉 - 黑色   唯品 会";
		String sen1 = " Ceaco   花色 纸质 二合一 拼图   3204 - 1 （ 1000 张 ）（ 美国 直发 ）  走秀 首页  >  生活  >  家庭装饰  >  装饰画  >  Ceaco   花色 纸质 二合一 拼图   3204 - 1 （ 1000 张 ）（ 美国 直发 ）  走秀网";
		String sen2 = "凝彩 眼线液 01 ( 黑色 )  3ml   首页  >  美妆 特卖  >  雅诗兰黛 EsteeLauder 化妆品 专场  >  凝彩 眼线液 01 ( 黑色 )  3ml   唯品 会";
		
		Vector<String> example= new Vector<String>();
		example.add(sen);
		example.add(sen1);
		example.add(sen2);
		
		
		
		Vector<String> result= nb.predict(example);
		Matrix r = nb.predictLogProba(example);
		System.out.println(result.get(0));
		System.out.println(result.get(1));
		System.out.println(result.get(2));
		for(int i=0;i<r.numRows();i++){
			for(int j=0;j<r.numColumns();j++){
				System.out.print(r.get(i, j) + " ");
			}
			System.out.println();
		}

		}





}

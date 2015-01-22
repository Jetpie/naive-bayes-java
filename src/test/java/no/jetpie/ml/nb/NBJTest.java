package no.jetpie.ml.nb;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.Vector;

import no.jetpie.ml.feature.Vectorizer;
import no.jetpie.ml.model.nb.MultinomialNB;
import no.jetpie.ml.model.nb.NaiveBayes;
import no.jetpie.ml.utils.Rule;
import no.uib.cipr.matrix.sparse.CompDiagMatrix;

import com.google.common.base.Charsets;
import com.google.common.io.Files;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonToken;

public class NBJTest {

	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
//      //Get the jvm heap size.
//      long heapSize = Runtime.getRuntime().totalMemory();
//
//      //Print the jvm heap size.
//      System.out.println("Heap Size = " + heapSize);
		
//		Vectorizer v = new Vectorizer("vocabulary.model",true);
//		NaiveBayes nb = new MultinomialNB(v,"proba/");
		String boundary =  "model_15_1_22/boundary.json";
		String model = "model_15_1_22/log_proba/";
		String vocabulary = "model_15_1_22/vocabulary.model";
//		Vectorizer v = new Vectorizer(vocabulary,true){
//			
//			@Override
//			protected void readVocab(String filepath) {
//
//				// initializationn
//				this.vocabulary = new HashMap<String, Integer>();
//				try {
//					File vocabModel = new File(filepath);
//					// guava read lines
//					List<String> lines = Files.readLines(vocabModel, Charsets.UTF_8);
//
//					// set number of vocabulary and idf values
//					this.numVocab = lines.size();
//					// init the diagonal value array for sparse diagonal matrix
//					this.idf = new CompDiagMatrix(this.numVocab, this.numVocab);
//					// iterate lines
//					for (String line : lines) {
//						String[] parts = line.split(",");
//						int ptr = Integer.parseInt(parts[1]);
//						this.vocabulary.put(parts[0], ptr);
//						this.idf.set(ptr, ptr, Double.parseDouble(parts[2]));
//					}
//				} catch (IOException e) {
//					e.printStackTrace();
//				}
//
//				System.out.println("vocabulary model imported..");
//			}
//		};
		
		NaiveBayes nb = new MultinomialNB(vocabulary,model,boundary);
		String sen = "贝 珍珠 恒美 绽放 耳钉 - 黑色   首页  >  Mbox 饰品 专场  >  贝 珍珠 恒美 绽放 耳钉 - 黑色   唯品 会";
		String sen1 = " Ceaco   花色 纸质 二合一 拼图   3204 - 1 （ 1000 张 ）（ 美国 直发 ）  走秀 首页  >  生活  >  家庭装饰  >  装饰画  >  Ceaco   花色 纸质 二合一 拼图   3204 - 1 （ 1000 张 ）（ 美国 直发 ）  走秀网";
		String sen2 = "凝彩 眼线液 01 ( 黑色 )  3ml   首页  >  美妆 特卖  >  雅诗兰黛 EsteeLauder 化妆品 专场  >  凝彩 眼线液 01 ( 黑色 )  3ml   唯品 会";
		String sen3 = "null";
		Vector<String> example= new Vector<String>();
		example.add(sen);
		example.add(sen1);
		example.add(sen2);
		example.add(sen3);
		example.add(sen2);
		example.add(sen1);
		example.add(sen);
		example.add(sen2);
		
		
		
		nb.setRule(1, new Rule("model/rules/mogujie.txt"));
		nb.setRule(2, new Rule("model/rules/meilishuo.txt"));
		int []states = {0,0,0,0,0,0,0,0};
		ArrayList<String> result= nb.predict(example,states);
	
		System.out.println(result.get(0));
		System.out.println(result.get(1));
		System.out.println(result.get(2));
		System.out.println(result.get(3));
		
//		Vector<String> oneDoc = new Vector<String>();
//		oneDoc.add(sen);
//		nb.predict(sen);
		
		
		
		String[] sens = {sen,sen1,sen2,sen3};
		Random r = new Random();
		r.setSeed(0);
		double ave = 0;
		System.out.println("---Read time testing---");
		for(int i=0;i<1000;i++){
			long startTime = System.nanoTime();
			nb.predict(sen,i%3);
			long endTime = System.nanoTime();
			long t = endTime - startTime;
			ave += t;
 			System.out.println("That took " +t  + " milliseconds");
			System.out.println();
		}
		System.out.println("average took :" + ave/(1000*1000000) + "ms");
		System.out.println("---End---");
		
		
		
		
//		System.out.println("---Read time testing for Batch---");
//		for(int i=0;i<100;i++){
//			long startTime = System.currentTimeMillis();
//			result= nb.predict(example);
//			long endTime = System.currentTimeMillis();
//			System.out.println("That took " + (endTime - startTime) + " milliseconds");
//			System.out.println();
//		}
//		System.out.println("---End---");

	}

}

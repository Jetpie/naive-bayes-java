package no.jetpie.ml.model.nb;

import java.io.File;
import java.io.FileReader;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.*;

//Matrix import
import no.uib.cipr.matrix.*;

//Guava import
import com.google.common.base.*;
import com.google.common.io.Files;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonToken;

import no.jetpie.ml.feature.Vectorizer;
/**
 * Base class for Naive Bayes Models
 * 
 * @author bingqingqu
 * @version 0.1.0
 * @date 2015.1.13
 *
 */
public abstract class NaiveBayes {

	/**
	 * transposed conditional probabilities matrix transposed form because
	 * later will be used as A * B^T
	 */
	protected Matrix TfeatureCondProb;
	/** prior : uniform distributed prior is applied */
	protected double logPrior;
	/** number of categories */
	protected int numCats;
	/** number of features */
	protected int numFeatures;
	/** category map */
	protected HashMap<Integer, String> category = new HashMap<Integer, String>();
	/** Vectorizer instance */
	protected Vectorizer vectorizer;
	/** threshold for each category */
	protected HashMap<String, Double> boundary = new HashMap<String, Double>();
	/** stop list for prediction */
	protected HashMap<String,Boolean> used = new HashMap<String,Boolean>();

	/**
	 * @param v
	 * 			A Vectorizer instance
	 * @param dirpath
	 * 			path to conditional probability directory
	 * @param boundary
	 * 			boundary json file
	 */
	public NaiveBayes(Vectorizer v, String dirpath, String boundary) {
		this.vectorizer = v;
		Preconditions.checkNotNull(this.vectorizer.getNumVocab(),
				"vocabulary model has no vocabulary!", this.vectorizer);
		this.numFeatures = this.vectorizer.getNumVocab();
		this.loadCondProba(dirpath);
		try {
			this.loadJsonFile(boundary);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		// uniform prior
		this.logPrior = 0 - Math.log((double) this.numCats);
	}

	/**
	 * @param vocabpath
	 * 			path to vocabulary model
	 * @param dirpath
	 * 			path to conditional probability directory
	 * @param boundary
	 * 			boundary json file
	 */
	public NaiveBayes(String vocabpath, String dirpath, String boundary) {
		this.vectorizer = new Vectorizer(vocabpath, true);
		Preconditions.checkNotNull(this.vectorizer.getNumVocab(),
				"vocabulary model has no vocabulary!", this.vectorizer);
		this.numFeatures = this.vectorizer.getNumVocab();
		this.loadCondProba(dirpath);
		try {
			this.loadJsonFile(boundary);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		// uniform prior
		this.logPrior = 0 - Math.log((double) this.numCats);
	}

	/**
	 * Read the conditional probabilities trained model
	 * 
	 * @param dirpath
	 *            path to the model DIRECTORY
	 */
	protected void loadCondProba(String dirpath){
		long startTime = System.currentTimeMillis();
		// main body
		File rootDir = new File(dirpath);
		Preconditions.checkState(rootDir.isDirectory(),
				"Path is not a directory!", rootDir);

		// current just check extension with ".txt"
		File [] files = rootDir.listFiles(new FilenameFilter() {
			public boolean accept(File dir, String name) {
				if (name.lastIndexOf('.') > 0) {
					// get last index for '.' char
					int lastIndex = name.lastIndexOf('.');
					// get extension
					String ext = name.substring(lastIndex);
					// match path name extension
					if (ext.equals(".txt")) {
						return true;
					}
				}
				return false;
			}
		});
		this.numCats = files.length;
		
		// sort the category in descending order
		Collections.sort(Arrays.asList(files));
		double[][] values = new double[this.numCats][this.numFeatures];
		int ptr = 0;
		for (File file : files) {
			if (file.isFile()) {
				// guava read lines
				try {
					String[] names = file.getName().split("\\.");
					this.category.put(ptr, names[0]);

					List<String> lines = Files.readLines(file, Charsets.UTF_8);
					for (String line : lines) {
						String[] parts = line.split(":");
						values[ptr][this.vectorizer.getPosInCol(parts[0])] = Double
								.parseDouble(parts[1]);
					}
					ptr++;
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		this.TfeatureCondProb = new DenseMatrix(this.numFeatures, this.numCats);
		new DenseMatrix(values).transpose(this.TfeatureCondProb);
		System.out.print("conditional probabilities parameters loaded..");
		System.out.println("(Elasped time: " + 
		(System.currentTimeMillis()-startTime)/1000 + "s)");
	}

	/**
	 * 
	 * @param jsonfile
	 * 			json file contains: 1. threshold and 2. stop flag
	 * @throws IOException 
	 */
	protected void loadJsonFile(String jsonfile) throws IOException{
		// TODO write another read json for online
		long startTime = System.currentTimeMillis();
		JsonReader jsonReader = new JsonReader(new FileReader(jsonfile));
		jsonReader.beginObject();
		// no safty here
		String key = null;
		while(jsonReader.hasNext()){
			switch(jsonReader.peek()){
			case NAME:
				key = jsonReader.nextName();
			case BEGIN_OBJECT:
				jsonReader.beginObject();
				jsonReader.nextName();
				this.boundary.put(key,jsonReader.nextDouble());
				jsonReader.nextName();
				this.used.put(key,jsonReader.nextInt()!=0);
			case END_OBJECT:
				jsonReader.endObject();
			default:
				break;	
			}
		}
		jsonReader.endObject();
		jsonReader.close();
		for(String k:this.boundary.keySet()){
			System.out.println(k + " " + this.boundary.get(k));
		}
		System.out.print("threshold and stop list loaded..");
		System.out.println("(Elasped time: " + 
		(System.currentTimeMillis()-startTime) + "ms)");
	}
	
	
	/**
	 * return the prior*likelihood (joint likelihood) result according to the
	 * input matrix
	 * 
	 * @param X
	 * 			Input document-feature matrix
	 * @return
	 * 			LOG joint likelihood document-category matrix
	 */
	abstract protected Matrix jointLogLikelihood(Matrix X);

	/**
	 * 
	 * @param documents
	 * 			List of documents of tokens separated by whitespace
	 * @return
	 * 			List of category predictions
	 */
	public ArrayList<String> predict(List<String> documents) {
		
		// transform documents into feature vectors
		Matrix X = this.vectorizer.transform(documents);
		// get joint log likelihood 
		Matrix jil = jointLogLikelihood(X);
		// normalize using log exp sum
		this.rowLogNormalize(jil);
		int[] predictions = this.argmax(jil);
		ArrayList<String> labels = new ArrayList<String>();

		for (int i = 0; i < predictions.length; i++) {
			String pred_cate = this.category.get(predictions[i]);
			if(jil.get(i, predictions[i]) > this.boundary.get(pred_cate))
				labels.add(pred_cate);
			else
				labels.add(null);
		}
		return labels;
	}
	
	/**
	 * 
	 * @param document
	 * 			Single document of tokens separated by whitespace
	 * @return predicted category
	 * 
	 */
	public String predict(String document) {
		//TODO improve the efficiency by using vector later
		
		long startTime = System.currentTimeMillis();
		ArrayList<String> documents = new ArrayList<String>();
		documents.add(document);
		long endTime = System.currentTimeMillis();
		System.out.println("vector add " + (endTime - startTime) + " milliseconds");
		
		
		startTime = System.currentTimeMillis();
		Matrix X = this.vectorizer.transform(documents);
		endTime = System.currentTimeMillis();
		System.out.println("transform " + (endTime - startTime) + " milliseconds");
		
		
		startTime = System.currentTimeMillis();
		Matrix jil = jointLogLikelihood(X);
		this.rowLogNormalize(jil);
		int[] predictions = this.argmax(jil);
		// return predictions
		String pred_cate = this.category.get(predictions[0]);
		endTime = System.currentTimeMillis();
		System.out.println("prediction " + (endTime - startTime) + " milliseconds");
		
		if(jil.get(0, predictions[0]) > this.boundary.get(pred_cate))
			return pred_cate;
		else
			return null;
	}

	/**
	 * 
	 * @param documents
	 * 			documents of terms separated by white space
	 * @return LOG joint likelihood for each document-category 
	 * pair 	 
	 */
	public Matrix predictLogProba(ArrayList<String> documents) {
		Matrix X = this.vectorizer.transform(documents);
		Matrix jil = this.jointLogLikelihood(X);
		this.rowLogNormalize(jil);
		return jil;
	}

	/**
	 * 
	 * @param document
	 * 			Single document of terms separated by whitespace
	 * @return
	 * 			LOG joint likelihood vector
	 */
	public Matrix predictLogProba(String document) {

		ArrayList<String> documents = new ArrayList<String>();
		documents.add(document);
		Matrix X = this.vectorizer.transform(documents);
		Matrix jil = this.jointLogLikelihood(X);
		this.rowLogNormalize(jil);
		return jil;
	}

	/**
	 * 
	 * @param documents of terms separated by whitespace
	 * @return joint likelihood of documents
	 */
	public Matrix predictProba(ArrayList<String> documents){
		Matrix jil = this.predictLogProba(documents);
		for(MatrixEntry e: jil){
			e.set(Math.exp(e.get()));
		}
		return jil;
	}
	
	/**
	 * 
	 * @param document of terms separated by whitespace
	 * @return joint likelihood of documents
	 */
	public Matrix predictProba(String document){
		Matrix jil = this.predictLogProba(document);
		for(MatrixEntry e: jil){
			e.set(Math.exp(e.get()));
		}
		return jil;
	}
	
	
	/**
	 * 
	 * @param jil
	 *            nDocs * nCats Matrix of predictions for each doc and cat pairs
	 * @return size = nDocs integer array for predictions
	 */
	private int[] argmax(Matrix jil) {
		int[] predictions = new int[jil.numRows()];
		for (int i = 0; i < jil.numRows(); i++) {
			int curPred = -1;
			// as logarithm for negative
			double curMax = Double.NEGATIVE_INFINITY;
			for (int j = 0; j < jil.numColumns(); j++) {
				if (jil.get(i, j) > curMax && jil.get(i, j) != 0) {
					curPred = j;
					curMax = jil.get(i, j);
				}
			}
			predictions[i] = curPred;
		}
		return predictions;
	}

	
	/**
	 * Normalize the Matrix by row log(sum(exp(X)))
	 * 
	 * @param jil
	 *            joint likelihood matrix for each category
	 */
	private void rowLogNormalize(Matrix jil) {
		// get max values for each row
		double[] rowMax = new double[jil.numRows()];
		Arrays.fill(rowMax, Double.NEGATIVE_INFINITY);
		for (MatrixEntry e : jil) {
			if (e.get() > rowMax[e.row()])
				rowMax[e.row()] = e.get();
		}
		// get log(sum(exp(row))) for each row
		// caution: log is a natural log here
		double[] logsumexp = new double[jil.numRows()];
		for (MatrixEntry e : jil) {
			logsumexp[e.row()] += Math.exp(e.get() - rowMax[e.row()]);
		}
		for (int i = 0; i < logsumexp.length; i++) {
			logsumexp[i] = Math.log(logsumexp[i]);
			logsumexp[i] += rowMax[i];
		}

		// normalize by divide
		// minus instead of divide in log domain
		for (MatrixEntry e : jil) {
			e.set(e.get() - logsumexp[e.row()]);
		}

		return;
	}
}
package no.jetpie.ml.model.nb;

import java.io.File;
import java.io.FileReader;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import com.google.common.base.Charsets;
import com.google.common.base.Preconditions;
import com.google.common.io.Files;
import com.google.gson.stream.JsonReader;

import no.uib.cipr.matrix.*;
import no.jetpie.ml.feature.TfidfVectorizer;
import no.jetpie.ml.feature.Vectorizer;
/**
 * The multinomial Naive Bayes model
 * 
 * @author bingqingqu
 * @version 0.1.2
 * @date 2015.1.13
 *
 */
public class MultinomialNB extends NaiveBayes{
	
	/** conditional probability model file suffix */
	private final String FILE_SUFFIX = ".txt";
	
	public MultinomialNB(Vectorizer v,String filePath, String thresholdPath){
		super(v, filePath, thresholdPath);
	}
	
	public MultinomialNB(String vocabPath,String filePath, String thresholdPath){
		super(vocabPath, filePath, thresholdPath);
	}	
	
	/**
	 * Read the conditional probabilities trained model
	 * 
	 * @param dirPath
	 * 		path to the model DIRECTORY
	 */
	protected void loadCondProba(String dirPath){
		long startTime = System.currentTimeMillis();
		// main body
		File rootDir = new File(dirPath);
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
					if (ext.equals(FILE_SUFFIX)) {
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
	 * 		json file contains: 1. threshold and 2. stop flag
	 * @throws IOException 
	 */
	protected void loadJsonFile(String jsonfile) throws IOException{
		
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
				this.threshold.put(key,jsonReader.nextDouble());
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
		
		System.out.print("threshold and stop list loaded..");
		System.out.println("(Elasped time: " + 
		(System.currentTimeMillis()-startTime) + "ms)");
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

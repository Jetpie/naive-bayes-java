package no.jetpie.ml.feature;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;

import com.google.common.base.Charsets;
import com.google.common.io.Files;

import no.uib.cipr.matrix.Matrix;
import no.uib.cipr.matrix.MatrixEntry;

public class TfidfVectorizer extends Vectorizer{
	
	/** flag control to use smooth on idf */
	protected boolean useIdf;
//	/** idf as a diagonal compressed(sparse) matrix */
//	protected CompDiagMatrix idf;
	
	/** idf square matrix diagonal values 
	 *  this is optimized to use an array
	 *  rather than an MTJ CompDiagMatrix
	 */
	protected double [] idfDiag;

	/**
	 * 
	 * @param filepath
	 * 		path to vocabulary model
	 * @param useIdf
	 * 		set true if idf prior will be applied
	 */
	public TfidfVectorizer(String filePath, boolean useIdf) {

		super(filePath);
		// import vocabulary
		this.useIdf = useIdf;
	}
	
	/**
	 * import vocabulary model from file
	 * 
	 * @param filePath
	 * 		path to model file
	 */
	public void readVocab(String filePath) {
		long startTime = System.currentTimeMillis();
		
		// initialization
		this.vocabulary = new HashMap<String, Integer>();
		try {
			File vocabModel = new File(filePath);
			// guava read lines
			List<String> lines = Files.readLines(vocabModel, Charsets.UTF_8);

			// set number of vocabulary and idf values
			this.numVocab = lines.size();
			// init the diagonal value array for sparse diagonal matrix
			this.idfDiag = new double[this.numVocab];
			// iterate lines
			for (String line : lines) {
				String[] parts = line.split(",");
				int ptr = Integer.parseInt(parts[1]);
				this.vocabulary.put(parts[0], ptr);
				this.idfDiag[ptr] = Double.parseDouble(parts[2]);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

		System.out.print("vocabulary model imported..");
		long endTime = System.currentTimeMillis();
		System.out.println("(Elasped time: " + (endTime-startTime) + " ms)");
	}
	
	/**
	 * Optimized method for matrix and diagonal matrix multiplication
	 * 
	 * @param documents
	 * 		A List of documents
	 * @return sparse input matrix
	 */
	public Matrix transform(List<String> documents) {
		Matrix X = this.countVocab(documents);
		
		if (this.useIdf) {
			// theoretically, use X * idf_diagonal should be faster.
			// unfortunately, the writer of mtj doesn't optimize it.
			// I use a manully optimization to make it faster.
			for(MatrixEntry e: X){
				e.set(e.get() * this.idfDiag[e.column()]);
			}
		} 
		// normalization
		this.rowNormalize(X);
		return X;
	}
}

package no.jetpie.ml.feature;

import java.io.*;
import java.util.*;

// Matrix import 
import no.uib.cipr.matrix.*;
import no.uib.cipr.matrix.sparse.*;

// Guava import
import com.google.common.base.*;
import com.google.common.io.Files;
import com.google.common.collect.*;

/**
 * Implementation with respect to python scikit-learn 
 * Special thanks to GUAVA
 * 
 * @author bingqingqu
 * @version 0.1.0
 * @date 2015.1.13
 *
 */
public class Vectorizer {

	/** number of vocabulary */
	protected int numVocab;
	/** flag control to use smooth on idf */
	private boolean useIdf;
	/** set n-gram wording strategy */
	private int[] N_GRAM = { 1, 2 };
	/** keywords and its index in row */
	protected HashMap<String, Integer> vocabulary;
//	/** idf as a diagonal compressed(sparse) matrix */
//	protected CompDiagMatrix idf;
	
	/** idf square matrix diagonal values 
	 *  this is optimized to use an array
	 *  rather than an MTJ CompDiagMatrix
	 */
	protected double [] idfDiag;

	/** guava splitter */
	private Splitter g_splitter = Splitter.on(" ");

	
	/**
	 * 
	 * @param filepath
	 * 			path to vocabulary model
	 * @param useIdf
	 * 			set true if idf prior will be applied
	 */
	public Vectorizer(String filepath, boolean useIdf) {

		// import vocabulary
		this.readVocab(filepath);
		this.useIdf = useIdf;
		
		// check if vocabulary is valid (guava)
		Preconditions.checkState(!this.vocabulary.isEmpty(),
				"TfidfVectorizer.vocabulary is empty!", this.vocabulary);
	}

	/**
	 * import vocabulary model from file
	 * 
	 * @param filepath
	 *            path to model file
	 */
	protected void readVocab(String filepath) {
		long startTime = System.currentTimeMillis();
		
		// initialization
		this.vocabulary = new HashMap<String, Integer>();
		try {
			File vocabModel = new File(filepath);
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
	 * count input documents and return sparse feature tf matrix
	 * 
	 * @param documents
	 * 			a set of documents of terms separated by terms
	 * @return Sparse matrix of keywords counts for each documents
	 */
	private Matrix countVocab(List<String> documents) {
		
		int numOfDocs = documents.size();
		// nz parameter for initialization sparse matrix in MTJ
		int[][] nz = new int[numOfDocs][];
		int[][] values = new int[numOfDocs][];
		
		// start to construct the sparse matrix information
		// row pointer
		int row = 0;
		for (String document : documents) {

			// tokenize the document
			LinkedList<String> terms = this.tokenize(document);
			Multiset<String> countTerms = HashMultiset.create(terms);
			// Retains only the elements in this collection that are 
			// contained in the vocabulary keyset
			countTerms.retainAll((Collection<String>) this.vocabulary.keySet());
			// initialize the current row index and respected value
			int[] curRow = new int[countTerms.elementSet().size()];
			int[] rowVal = new int[countTerms.elementSet().size()];
			int i = 0;
			for (String term : countTerms.elementSet()) {

				curRow[i] = this.vocabulary.get(term);
				rowVal[i++] = countTerms.count(term);
			}
			nz[row] = curRow;
			values[row++] = rowVal;
		}
		// initialize the sparse matrix
		Matrix X = new CompRowMatrix(numOfDocs, this.numVocab, nz);
		for (int i = 0; i < nz.length; i++) {
			for (int j = 0; j < nz[i].length; j++) {
				X.set(i, nz[i][j], values[i][j]);
			}
		}
		return X;
	}
	
	/**
	 * 
	 * @return size of vocabulary
	 */
	public int getNumVocab() {
		return this.numVocab;
	}

	/**
	 * 
	 * @param term
	 * 			one of keywords
	 * @return column position of the term
	 * 
	 */
	public int getPosInCol(String term) {
		int pos = this.vocabulary.get(term);
		Preconditions.checkNotNull(pos,
				"vocabulary and likelihood model matching problem!", term);
		return pos;
	}

	/**
	 * tokenize the input stream N_GRAM
	 * 
	 * @param document
	 *            a document of terms separeted by whitespace
	 * @return a List of String satisfied n-gram
	 */
	private LinkedList<String> tokenize(String document) {

		// length limitations by n-gram
		int MIN_N = this.N_GRAM[0];
		int MAX_N = this.N_GRAM[1];
		Preconditions.checkState(MIN_N <= MAX_N,
				"n-gram settings wrong! min < max");
		// split using guava
		String[] terms = Iterables.toArray(g_splitter.trimResults()
				.omitEmptyStrings().split(document), String.class);
		// terminate if 1-gram is needed
		if (MAX_N == 1)
			return new LinkedList<String>(Arrays.asList(terms));
		Preconditions.checkState(MIN_N < MAX_N,
				"n-gram settings wrong! min = max != 1");
		if (MAX_N > 1 && terms.length > 0) {
			LinkedList<String> tokens = new LinkedList<String>();
			for (int n = MIN_N; n <= MAX_N; n++) {
				for (int i = 0; i < terms.length - n + 1; i++) {
					StringBuilder sb = new StringBuilder();
					for (int j = i; j < i + n; j++)
						sb.append((j > i ? " " : "") + terms[j]);
					tokens.add(sb.toString());
				}
			}
			return tokens;
		}
		return null;
	}


	@Deprecated
	private void fit() {
	}

	
	/**
	 * Optimized method for matrix and diagonal matrix multiplication
	 * 
	 * @param documents
	 * 			A List of documents
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

	/**
	 * Currently not implemented for online
	 * 
	 * @return
	 * 			the trained vectorizer
	 */
	public Vectorizer fitTransform() {
		return this;
	}

	
	/**
	 * normalize matrix by row using L2-norm
	 * 
	 * @param X
	 * 			the row-normalized matrix
	 */
	private void rowNormalize(Matrix X) {
		//init
		double [] rowRootSum = new double[X.numRows()];
		// sum
		for(MatrixEntry e: X){
			rowRootSum[e.row()] += Math.pow(e.get(),2);
		}
		// root
		for(int i=0;i <rowRootSum.length;i++){
			rowRootSum[i] = Math.sqrt(rowRootSum[i]);
		}
		// normalize
		for(MatrixEntry e: X){
			e.set(e.get()/rowRootSum[e.row()]);
		}
	}

}

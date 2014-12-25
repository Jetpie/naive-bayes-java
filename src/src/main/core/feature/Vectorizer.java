package src.main.core.feature;

import java.io.*;
import java.util.*;
import java.util.Vector;
import java.util.concurrent.TimeUnit;


// Matrix import 
import no.uib.cipr.matrix.*;
import no.uib.cipr.matrix.sparse.*;

// Guava import
import com.google.common.base.*;
import com.google.common.io.Files;
import com.google.common.collect.*;

/**
 * Implementation in terms of python scikit-learn 
 * Special thanks to GUAVA
 * 
 * @author bingqingqu
 * @version 0.1.0
 * @date 2014.12.08
 *
 */
public class Vectorizer {

	/** number of vocabulary */
	private int numVocab;
	/** flag control to use smooth on idf */
	private boolean useIdf;
	/** flag control to use normalization */
	private boolean norm;
	/** set n-gram wording strategy */
	private int[] N_GRAM = { 1, 2 };
	/** keywords and its index in row */
	private HashMap<String, Integer> vocabulary;

	/** idf as a diagonal compressed(sparse) matrix */
	private CompDiagMatrix idf;

	/** stopwatch util for time counting */
	private Stopwatch stopwatch = Stopwatch.createStarted();

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
	}

	/**
	 * import vocabulary model from file
	 * 
	 * @param filepath
	 *            path to model file
	 */
	private void readVocab(String filepath) {
		// reset the stopwatch
		stopwatch.reset();
		stopwatch.start();
		// initializationn
		this.vocabulary = new HashMap<String, Integer>();
		try {
			File vocabModel = new File(filepath);
			// guava read lines
			List<String> lines = Files.readLines(vocabModel, Charsets.UTF_8);

			// set number of vocabulary and idf values
			this.numVocab = lines.size();
			// init the diagonal value array for sparse diagonal matrix
			this.idf = new CompDiagMatrix(this.numVocab, this.numVocab);
			// iterate lines
			for (String line : lines) {
				String[] parts = line.split(",");
				int ptr = Integer.parseInt(parts[1]);
				this.vocabulary.put(parts[0], ptr);
				this.idf.set(ptr, ptr, Double.parseDouble(parts[2]));
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

		System.out.print("vocabulary model imported..");
		long millis = stopwatch.elapsed(TimeUnit.MILLISECONDS);
		System.out.println("(Elasped time: " + millis + " ms)");
	}

	/**
	 * count input documents and return sparse feature tf vectors
	 * 
	 * @param documents
	 * 			a set of documents of terms separated by terms
	 * @return Sparse matrix of keywords counts for each documents
	 */
	private Matrix countVocab(Vector<String> documents) {
		// check if vocabulary is valid (guava)
		Preconditions.checkState(!this.vocabulary.isEmpty(),
				"TfidfVectorizer.vocabulary is empty!", this.vocabulary);

		int numOfDocs = documents.size();
		// nz parameter for initialization sparse matrix in MTJ
		int[][] nz = new int[numOfDocs][];
		int[][] values = new int[numOfDocs][];
		// row pointer
		int row = 0;
		for (String document : documents) {

			// tokenize the document
			Vector<String> terms = this.tokenize(document);
			Multiset<String> countTerms = HashMultiset.create(terms);
			countTerms.retainAll((Collection<String>) this.vocabulary.keySet());
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
	 * @return a vector of String satisfied n-gram
	 */
	private Vector<String> tokenize(String document) {

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
			return new Vector<String>(Arrays.asList(terms));
		Preconditions.checkState(MIN_N < MAX_N,
				"n-gram settings wrong! min = max != 1");
		if (MAX_N > 1 && terms.length > 0) {
			Vector<String> tokens = new Vector<String>();
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
	 * Outer method to transform input documents
	 * 
	 * @param documents
	 * 			A vector of documents
	 * @return sparse input matrix
	 */
	public Matrix transform(Vector<String> documents) {
		Matrix X = this.countVocab(documents);
		// System.out.println("row: " + X.numRows() + " cols: " + X.numColumns()
		// );
		// initializ tfidf matrix with same parameters of X
		Matrix tfidf;
		if (X instanceof CompRowMatrix) {
			tfidf = new CompRowMatrix(X);
		} else {
			tfidf = new DenseMatrix(X);
		}
		if (this.useIdf) {
			// multiply tf with idf diagonal matrix
			// return tfidf document-term matrix
			// caution: unnormalize until now
			X.mult(this.idf, tfidf);
			this.rowNormalize(tfidf);
			return tfidf;
		} else
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

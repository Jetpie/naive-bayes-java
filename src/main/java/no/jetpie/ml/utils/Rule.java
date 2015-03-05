package no.jetpie.ml.utils;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import com.google.common.base.Charsets;
import com.google.common.io.Files;

/**
 * make constraints on categories for specify merchants/brands or so
 * 
 * @author bingqingqu
 * @version 0.1.2
 * @date 2015.1.21
 *
 */
public class Rule {
	/** the rule for prediction constraint */
	private Set<String> constraint;
	
	public Rule(String filepath){
		this.loadRule(filepath);
	}
	
	/**
	 * 
	 * @param filepath
	 * 		path to the rule file
	 */
	public void loadRule(String filepath){
		// initialization
		this.constraint = new HashSet<String>();
		try {
			File ruleFile = new File(filepath);
			// guava read lines
			List<String> lines = Files.readLines(ruleFile, Charsets.UTF_8);

			// iterate lines
			for (String line : lines) {
				this.constraint.add(line);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * 
	 * @return the if the current category is possibly a valid predictions
	 * 
	 */
	public boolean isValid(String category){
		return this.constraint.contains(category);
	}
	
}

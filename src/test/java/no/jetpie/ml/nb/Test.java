package no.jetpie.ml.nb;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;

import com.google.common.base.CharMatcher;
import com.google.common.base.Predicate;
import com.google.common.base.Splitter;
import com.google.common.collect.Iterables;

public class Test {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Splitter g_splitter = Splitter.on(" ");
		Iterable<String> results = g_splitter.trimResults()
				.split("AB ,  B   CS 你好    你 好 你好".toLowerCase());
		Predicate<String> noSingleLetter = new Predicate<String>(){

			public boolean apply(String input) {
				// TODO Auto-generated method stub
				return input.length()>1;
			}
			
		};
		ArrayList<String> candidates = new ArrayList<String>();
		for(String t : Iterables.filter(results,noSingleLetter)){
			System.out.println(t);
		}
		
		String[] terms = (String[]) candidates.toArray();
		LinkedList<String> tokens = new LinkedList<String>();
		for (int n = 1; n <= 2; n++) {
			for (int i = 0; i < terms.length - n + 1; i++) {
				StringBuilder sb = new StringBuilder();
				for (int j = i; j < i + n; j++)
					sb.append((j > i ? " " : "") + terms[j]);
				tokens.add(sb.toString());
			}
		}
		
		System.out.println(Arrays.toString(tokens.toArray()));
	}

}

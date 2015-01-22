package no.jetpie.ml.nb;

import java.util.HashMap;

public class Test {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		HashMap<Integer,Boolean> states = new HashMap<Integer,Boolean>();
		states.put(1, true);
		states.put(2, false);
		if(states.get(3)){
			System.out.println("not null");
		}

	}

}

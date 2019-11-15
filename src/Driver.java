import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Enumeration;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.lazy.KStar;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class Driver {
	public static Instances getDataSet(String fileName) throws IOException {
		int classIdx = 1;
		ArffLoader loader = new ArffLoader();
		loader.setSource(Driver.class.getResourceAsStream(fileName));
		Instances dataSet = loader.getDataSet();
		dataSet.setClassIndex(classIdx);
		return dataSet;
	}

 
	public static void main(String[] args) throws Exception {
		Instances data = getDataSet("car_data.arff");
		data.setClassIndex(data.numAttributes() - 1);
		
		Classifier zr = new ZeroR();		
		zr.buildClassifier(data);
		System.out.println(zr);
        Evaluation evalz = new Evaluation(data);
        evalz.evaluateModel(zr, data);
		System.out.println("** KNN ZeroR **");
		System.out.println(evalz.toSummaryString());
	    System.out.println(evalz.toClassDetailsString());
		System.out.println(evalz.toMatrixString());
		
		Classifier ibk1 = new IBk(1);		
		ibk1.buildClassifier(data);
		System.out.println(ibk1);
        Evaluation evali1 = new Evaluation(data);
        evali1.evaluateModel(ibk1, data);
		System.out.println("** KNN IBK K = 1 **");
		System.out.println(evali1.toSummaryString());
	    System.out.println(evali1.toClassDetailsString());
		System.out.println(evali1.toMatrixString());
		
		Classifier ibk2 = new IBk(2);		
		ibk2.buildClassifier(data);
		System.out.println(ibk2);
        Evaluation evali2 = new Evaluation(data);
        evali2.evaluateModel(ibk2, data);
		System.out.println("** KNN IBK K = 2 **");
		System.out.println(evali2.toSummaryString());
	    System.out.println(evali2.toClassDetailsString());
		System.out.println(evali2.toMatrixString());
		
		Classifier ibk3 = new IBk(3);		
		ibk3.buildClassifier(data);
		System.out.println(ibk3);
        Evaluation evali3 = new Evaluation(data);
        evali3.evaluateModel(ibk3, data);
		System.out.println("** KNN IBK K = 3 **");
		System.out.println(evali3.toSummaryString());
	    System.out.println(evali3.toClassDetailsString());
		System.out.println(evali3.toMatrixString());
		
		Classifier kstar = new KStar();		
		kstar.buildClassifier(data);
		System.out.println(kstar);
        Evaluation evalk = new Evaluation(data);
        evalk.evaluateModel(kstar, data);
		System.out.println("** KNN KStar **");
		System.out.println(evalk.toSummaryString());
	    System.out.println(evalk.toClassDetailsString());
		System.out.println(evalk.toMatrixString());
		
		/*
		Classifier j48 = new J48();		
		String[] options = new String[1];
		options[0]="-U";
		j48.buildClassifier(data);
		System.out.println(j48);
        Evaluation evalj = new Evaluation(data);
        evalk.evaluateModel(j48, data);
		System.out.println("** KNN J48 **");
		System.out.println(evalj.toSummaryString());
	    System.out.println(evalj.toClassDetailsString());
		System.out.println(evalj.toMatrixString());
		*/
		
	}
}

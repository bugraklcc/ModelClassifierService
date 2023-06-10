package com.example.model.Controller;

import weka.classifiers.bayes.NaiveBayes;

import java.util.HashMap;
import java.util.Map;

import org.springframework.stereotype.Component;

@Component
public class ModelManager {

    private Map<String, NaiveBayes> models;

    public ModelManager() {
        models = new HashMap<>();
    }

    public void addModel(String string, NaiveBayes model) {
        models.put(string, model);
    }

    public NaiveBayes getModel(String string) {
        return models.get(string);
    }

}

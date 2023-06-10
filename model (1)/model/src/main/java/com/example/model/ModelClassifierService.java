package com.example.model;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.netflix.eureka.server.EnableEurekaServer;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.tokenizers.NGramTokenizer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.File;
import java.util.ArrayList;

@EnableEurekaServer
@SpringBootApplication
public class ModelClassifierService {

    public static void main(String[] args) throws Exception {
        SpringApplication.run(ModelClassifierService.class, args);
    }

    private static NaiveBayes loadModel1() {
        try {
            File modelFile = new File("model/models/model1.model");
            NaiveBayes model1 = (NaiveBayes) SerializationHelper.read(modelFile.getAbsolutePath());
            return model1;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private static NaiveBayes loadModel2() {
        try {
            File modelFile = new File("model/models/model2.model");
            NaiveBayes model2 = (NaiveBayes) SerializationHelper.read(modelFile.getAbsolutePath());
            return model2;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    @RestController
    @RequestMapping("/classify")
    public class ClassifyController {

        public ClassifyController() {
        }

        @PostMapping("/model1")
        public String classifyModel1(@RequestBody String htmlContent) {
            String extractedText = extractTextFromHtml(htmlContent);
            NaiveBayes model1 = loadModel1();

            try {
                // Maili sınıflandır
                double[] probabilities = model1.distributionForInstance(createInstance(htmlContent));

                // En yüksek olasılığa sahip sınıfı bul
                int predictedClass = maxIndex(probabilities);

                // Normal tahmini yapıldıysa, 2. modele yönlendir
                if (predictedClass == 1) {
                    return classifyModel2(extractedText);
                } else {
                    // Spam tahmini yapıldıysa, spam olarak işaretle
                    return "Spam";
                }
            } catch (Exception e) {
                e.printStackTrace();
                return "Error occurred during classification.";
            }
        }

        @PostMapping("/model2")
        public String classifyModel2(@RequestBody String htmlContent) {
            
            NaiveBayes model2 = loadModel2();

            try {
                // Maili sınıflandır
                double[] probabilities = model2.distributionForInstance(createInstance(htmlContent));

                // Tahmin sonucunu döndür
                return getPredictedClass(probabilities);
            } catch (Exception e) {
                e.printStackTrace();
                return "Error occurred during classification.";
            }
        }

        private Instance createInstance(String htmlContent) throws Exception {
            // Veri kümesini oluştur
            String relationName = "normspam";
            ArrayList<Attribute> attributes = new ArrayList<>();

            // Öznitelikleri tanımla
            Attribute textAttribute = new Attribute("text", (ArrayList<String>) null);
            attributes.add(textAttribute);

            // Sınıfları tanımla
            ArrayList<String> classValues = new ArrayList<>();
            classValues.add("birincil");
            classValues.add("guncellemeler");
            classValues.add("tanitimlar");
            classValues.add("sosyal");
            Attribute classAttribute = new Attribute("@@class@@", classValues);
            attributes.add(classAttribute);

            // Atributeleri diziye çevir
            Attribute[] attributeArray = new Attribute[attributes.size()];
            attributeArray = attributes.toArray(attributeArray);

            // Veri kümesini oluştur ve sınıf indeksini ayarla
            Instances dataset = new Instances(relationName, attributes, 0);
            dataset.setClassIndex(1); // Sınıf indeksi 1 olarak ayarlandı

            // Mail içeriğini özniteliklere dönüştür
            DenseInstance instance = new DenseInstance(1.0, new double[attributeArray.length]);
            instance.setValue(attributeArray[0], extractTextFromHtml(htmlContent));

            // Veri kümesine örnek ekle
            dataset.add(instance);

            // Veri kümesini vektöre dönüştür
            Instances vectorizedInstance = applyStringToVectorFilter(dataset);

            return vectorizedInstance.get(0);
        }

       
        private String extractTextFromHtml(String htmlContent) {
            Document doc = Jsoup.parse(htmlContent);
            org.jsoup.select.Elements elements = doc.getAllElements();

            StringBuilder sb = new StringBuilder();
            for (Element element : elements) {
                String text = element.ownText();
                if (!text.isEmpty()) {
                    sb.append(text).append(" ");
                }
            }

            return sb.toString().trim();
        }

        private Instances applyStringToVectorFilter(Instances dataset) throws Exception {
            StringToWordVector stringToVector = new StringToWordVector();
            stringToVector.setTokenizer(new NGramTokenizer());
            stringToVector.setInputFormat(dataset);

            Instances vectorizedInstances = Filter.useFilter(dataset, stringToVector);
            vectorizedInstances.setClassIndex(0);

            return vectorizedInstances;
        }

        private int maxIndex(double[] probabilities) {
            int maxIndex = 0;
            double maxValue = probabilities[0];

            for (int i = 1; i < probabilities.length; i++) {
                if (probabilities[i] > maxValue) {
                    maxValue = probabilities[i];
                    maxIndex = i;
                }
            }

            return maxIndex;
        }

        private String getClassName(int predictedClass) {
            switch (predictedClass) {
                case 0:
                    return "birincil";
                case 1:
                    return "guncellemeler";
                case 2:
                    return "tanitimlar";
                case 3:
                    return "sosyal";
                default:
                    return "Unknown";
            }
        }

        private String getPredictedClass(double[] probabilities) {
            int maxIndex = maxIndex(probabilities);
            String predictedClass = getClassName(maxIndex);

            return predictedClass;
        }
    }
}

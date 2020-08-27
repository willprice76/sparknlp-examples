package org.example.sparknlp;

import com.johnsnowlabs.nlp.DocumentAssembler;
import com.johnsnowlabs.nlp.LightPipeline;
import com.johnsnowlabs.nlp.SparkNLP;
import com.johnsnowlabs.nlp.annotators.Tokenizer;
import com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknSentimentApproach;
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;
import scala.collection.JavaConversions;
import scala.collection.Seq;
import scala.collection.immutable.Map;

import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

@RestController
public class SparkNlpController {
    private static final Logger LOG = LoggerFactory.getLogger(SparkNlpController.class);
    private LightPipeline scoringPipeline;
    private SparkSession spark;

    public SparkNlpController() {
        spark = SparkNLP.start(false, false);
        scoringPipeline = new PretrainedPipeline("analyze_sentiment", "en").lightModel();
    }

    @PostMapping("/sentiment/score")
    public List<String> score(@RequestBody String[] inputData) {
        Instant start = Instant.now();
        LOG.debug("Analyzing {} rows of text data...", inputData.length);
        //Run test text through the pipeline
        Map<String, Seq<String>>[] annotations = scoringPipeline.annotate(inputData);
        //Mangle the sentiment data out
        List<String> output = Arrays.stream(annotations).map(annotation -> {
            List<String> sentiment = JavaConversions.seqAsJavaList(annotation.get("sentiment").getOrElse(null));
            return sentiment.get(0);
        }).collect(Collectors.toList());
        LOG.debug("Analysis completed in {} milliseconds", Duration.between(start, Instant.now()).toMillis());
        return output;
    }

    @PostMapping("/sentiment/train")
    public String train(@RequestBody List<TextData> data) throws IOException {
        Instant start = Instant.now();
        Dataset<Row> input = spark.createDataFrame(data, TextData.class);
        LOG.debug("Running training with {} rows of text data", data.size());
        Pipeline pipeline = getSentimentTrainingPipeline();
        PipelineModel newPipelineModel = pipeline.fit(input);
        long trainingTime = Duration.between(start, Instant.now()).toMillis();
        //Overwrite the existing scoring pipeline
        scoringPipeline = new LightPipeline(newPipelineModel, false);
        return String.format("Training completed in %s milliseconds", trainingTime);
    }

    private Pipeline getSentimentTrainingPipeline() {
        DocumentAssembler document = new DocumentAssembler();
        document.setInputCol("text");
        document.setOutputCol("document");

        String[] tokenizerInputCols = {"document"};
        Tokenizer tokenizer = new Tokenizer();
        tokenizer.setInputCols(tokenizerInputCols);
        tokenizer.setOutputCol("token");

        String[] sentimentInputCols = {"document", "token"};
        ViveknSentimentApproach sentimentApproach = new ViveknSentimentApproach();
        sentimentApproach.setInputCols(sentimentInputCols);
        sentimentApproach.setOutputCol("sentiment");
        sentimentApproach.setSentimentCol("label");
        sentimentApproach.setCorpusPrune(0);

        Pipeline pipeline = new Pipeline();
        pipeline.setStages(new PipelineStage[]{document, tokenizer, sentimentApproach});
        return pipeline;
    }
}
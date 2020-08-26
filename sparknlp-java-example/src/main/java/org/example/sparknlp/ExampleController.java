package org.example.sparknlp;

import com.johnsnowlabs.nlp.LightPipeline;
import com.johnsnowlabs.nlp.SparkNLP;
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;
import scala.collection.JavaConversions;
import scala.collection.Seq;
import scala.collection.immutable.Map;

import java.time.Duration;
import java.time.Instant;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

@RestController
public class ExampleController {
    private static final Logger LOG = LoggerFactory.getLogger(ExampleController.class);
    private LightPipeline scoringPipeline;

    public ExampleController() {
        SparkNLP.start(false, false);
        scoringPipeline = new PretrainedPipeline("analyze_sentiment", "en").lightModel();
    }

    @PostMapping("/analyze")
    public List<String> analyze(@RequestBody String[] inputData) {
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
}

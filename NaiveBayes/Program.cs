using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;
using System.Threading;

namespace NaiveBayes
{
    class MainClass
    {
        public static void Main(string[] args)
        {
            //PARSER FOR http://inclass.kaggle.com/c/si650winter11/data DATASET
            string inputText = File.ReadAllText("in.csv");
            List<string> positiveSet = new List<string>();
            List<string> negativeSet = new List<string>();

            //Console.WriteLine(nbModel.trainers.clean("The Da Vinci Code book is just awesome.".ToLower(), new char[] { ' ' }));
            Console.WriteLine("Started model");
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();

            string[] it = inputText.Split('\n');
            //Array.Resize(ref it, 1000000);
            Console.WriteLine("Trimmed");
            Console.WriteLine(it.Length);
            for (int i = 1; i < it.Length; i++)
            {
                string entry = it[i];
                //if (i % 10000 == 0) Console.WriteLine((i/it.Length).ToString("0." + new string('#', 339)));//progressBar(i / it.Length, (int)Math.Round((double)Console.WindowWidth / 2));
                //Console.WriteLine(i);
                if (entry == "") continue;
                string[] columns = entry.Split(',');
                if (columns[1] == "0")
                {
                    negativeSet.Add(nbModel.trainers.clean(columns[3].ToLower(), new char[] { ' ' }));
                }
                else if (columns[1] == "1")
                {
                    positiveSet.Add(nbModel.trainers.clean(columns[3].ToLower(), new char[] { ' ' }));
                }
                else
                {
                    Console.WriteLine("Unknown sentiment '" + columns[1] + "'");
                }
            }

            stopwatch.Stop();
            Console.WriteLine("Done!");
            Console.WriteLine("Time elapsed: {0}", stopwatch.Elapsed);
            //END PARSER

            Console.WriteLine("Constructing model");
            Dictionary<string, List<string>> trainData = new Dictionary<string, List<string>>();
            trainData.Add("positive", positiveSet);
            trainData.Add("negative", negativeSet);
            nbModel.model textModel;
            textModel = nbModel.trainers.textToModel(trainData, ' ');
            //textModel.removeFeature("positive", "the");
            //textModel.removeFeature("negative", "the");
            Console.WriteLine("Done");
            Console.WriteLine("Testing model");
            nbModel.trainers.textTrainDataProvider tdp = new nbModel.trainers.textTrainDataProvider();
            tdp.regCatagory("positive", positiveSet, ' ');
            tdp.regCatagory("negative", negativeSet, ' ');
            double score = nbModel.trainers.testTextModel(textModel, tdp);
            //Console.WriteLine("Model got a {0}%", score.ToString("0." + new string('#', 339)));
            textModel = nbModel.trainers.trainTextToGoal(textModel, 75, tdp, false, false);
            Console.WriteLine("Model with {0} catagories", textModel.catagories.Count);
            while(true)
            {
                Console.Write("(enter a sentence)> ");
                string userInput = Console.ReadLine();
                List<object> t = nbModel.trainers.clean(userInput.ToLower(), new char[] { ' ' }).Split(' ').ToList<object>();

                Console.Clear();
                colorWriteComponents(userInput, textModel);

                Dictionary<object, double> pred = textModel.predict(t, true);
                Console.WriteLine("Components: " + predToString(pred));
                if (!nbModel.trainers.same(pred.Values.ToList())) Console.WriteLine("Prediction: " + pred.Keys.ElementAt(nbModel.trainers.max(pred)));
                else Console.WriteLine("Prediction: None");
            }
        }

        public static string predToString(Dictionary<object, double> pred)
        {
            string ret = "[";
            foreach (KeyValuePair<object, double> kvp in pred)
                ret += "{" + kvp.Key + ":" + kvp.Value.ToString("0." + new string('#', 339)) + "}";
            ret += "]";
            return ret;
        }

        public static void colorWriteComponents(string originalInput, nbModel.model model)
        {
            //The selectors
            string input = originalInput.ToLower();

            //string sentenceSelector = "[\"']?[A-Z][^.?!]+((?![.?!]['\"]?\\s[\"']?[A-Z][^.?!]).)+[.?!'\"]+"; //Strict punctuation
            string sentenceSelector = "[\"']?[A-Z][^.?!]+((?![.?!]['\"]?\\s[\"']?[A-Z][^.?!]).)."; //Allows for no puctuation
            string wordSelector = "(\\w+)";
            string punctuationSelector = "[.?!]";

            Dictionary<string, ConsoleColor> wordColors = new Dictionary<string, ConsoleColor>();

            //Get all the word colors
            List<string> words = System.Text.RegularExpressions.Regex.Matches(input, wordSelector).Cast<System.Text.RegularExpressions.Match>().Select((x)=>x.Value).Distinct().ToList();
            for (int i = 0; i < words.Count; i++)
            {
                Dictionary<object, double> pred = model.predict(new List<object> { ((object)words[i]) }, true);
                ConsoleColor color = ConsoleColor.White;
                if (!nbModel.trainers.same(pred.Values.ToList()))
                {
                    if ((string)pred.Keys.ElementAt(nbModel.trainers.max(pred)) == "positive")
                    {
                        color = ConsoleColor.DarkGreen;
                    }
                    else if ((string)pred.Keys.ElementAt(nbModel.trainers.max(pred)) == "negative")
                    {
                        color = ConsoleColor.DarkRed;
                    }
                }
                //else color = ConsoleColor.White;
                wordColors.Add(words[i], color);
            }

            System.Text.RegularExpressions.MatchCollection matches = System.Text.RegularExpressions.Regex.Matches(originalInput, sentenceSelector);
            for (int i = 0; i < matches.Count; i++)
            {
                //Get the words
                string sentence = matches[i].Value.ToLower();
                List<string> wordsFromSentence = System.Text.RegularExpressions.Regex.Matches(sentence, wordSelector).Cast<System.Text.RegularExpressions.Match>().Select((x) => x.Value).Distinct().ToList();

                Dictionary<object, double> pred = model.predict(new List<object> { ((object)words[i]) }, true);
                ConsoleColor backgroundColor = ConsoleColor.Black;
                if (!nbModel.trainers.same(pred.Values.ToList()))
                {
                    if ((string)pred.Keys.ElementAt(nbModel.trainers.max(pred)) == "positive")
                    {
                        backgroundColor = ConsoleColor.Green;
                    }
                    else if ((string)pred.Keys.ElementAt(nbModel.trainers.max(pred)) == "negative")
                    {
                        backgroundColor = ConsoleColor.Red;
                    }
                }

                Console.BackgroundColor = backgroundColor;

                for (int j = 0; j < wordsFromSentence.Count; j++)
                {
                    if (!wordColors.ContainsKey(wordsFromSentence[j]))
                    {
                        Console.ForegroundColor = ConsoleColor.DarkYellow;
                    }
                    else
                    {
                        Console.ForegroundColor = wordColors[wordsFromSentence[j]];
                    }

                    if (j == wordsFromSentence.Count - 1) Console.Write(wordsFromSentence[j]);
                    else Console.Write(wordsFromSentence[j] + " ");
                }

                Console.ForegroundColor = ConsoleColor.White;
                System.Text.RegularExpressions.Match match = System.Text.RegularExpressions.Regex.Match(sentence, punctuationSelector);
                if (match.Success) Console.Write(match.Value);
                Console.BackgroundColor = ConsoleColor.Black;
                Console.Write(" ");
            }

            Console.ResetColor();
            Console.WriteLine("");
        }
    }
}

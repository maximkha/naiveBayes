using System;
using System.Collections.Generic;
using System.Linq;

namespace NaiveBayes
{
    public static class nbModel
    {
        [Serializable]
        public class catagory
        {
            public object id;
            public int total;
            public Dictionary<object, int> frequency = new Dictionary<object, int>();
        }

        [Serializable]
        public class model
        {
            public List<catagory> catagories = new List<catagory>();

            public Dictionary<object, double> predict(List<object> features, bool ignoreFeature)
            {
                Dictionary<object, double> catagoryProb = new Dictionary<object, double>();
                foreach (catagory c in catagories)
                {
                    List<double> prob = new List<double>();
                    foreach (object feature in features)
                    {
                        if(!catagoryHasFeature(c,feature)) 
                        {
                            if (ignoreFeature)
                            {
                                prob.Add(1);
                            }
                            else
                            {
                                prob.Add(0);
                                break;
                            }
                        }
                        else prob.Add((double)c.frequency[feature]/(double)c.total);
                    }
                    catagoryProb.Add(c.id, multArray(prob));
                }

                return catagoryProb;
            }

            public Dictionary<object, double> predict(List<object> features)
            {
                return predict(features, false);
            }

            public bool catagoryHasFeature(catagory c, object feature)
            {
                return c.frequency.ContainsKey(feature);
            }

            public double multArray(List<double> vals)
            {
                double prev = 1;
                foreach (double value in vals)
                    prev *= value;
                return prev;
            }

            public void removeFeature(object catID, object featID)
            {
                int i = catagories.FindIndex((x) => x.id.Equals(catID));
                if (i == -1) return;
                if (!catagories[i].frequency.ContainsKey(featID)) return;
                catagories[i].total -= catagories[i].frequency[featID];
                catagories[i].frequency.Remove(featID);
            }

            public void removeFeature(object catID, object featID, int num)
            {
                int i = catagories.FindIndex((x) => x.id.Equals(catID));
                if (i == -1) return;
                if (!catagories[i].frequency.ContainsKey(featID)) return;
                if (catagories[i].frequency[featID]-num <= 0) {
                    removeFeature(catID, featID);
                } 
                else 
                {
                    catagories[i].frequency[featID] -= num;
                }
                catagories[i].total -= num;
            }

            public void addFeature(object catID, object featID, int num)
            {
                int i = catagories.FindIndex((x) => x.id.Equals(catID));
                if (i == -1) return;
                if (!catagories[i].frequency.ContainsKey(featID))
                    catagories[i].frequency.Add(featID, 0);
                catagories[i].total += num;
                catagories[i].frequency[featID] += num;
            }
        }

        public static class trainers
        {
            public static model textToModel(Dictionary<string,List<string>> catExample, char split)
            {
                model textModel = new model();

                for(int i = 0; i < catExample.Keys.Count; i++)
                {
                    catagory cat = new catagory();
                    foreach (string example in catExample.Values.ElementAt(i))
                    {
                        string[] parts = example.Split(split);
                        for (int j = 0; j < parts.Length; j++)
                        {
                            if (!cat.frequency.ContainsKey(parts[j])) cat.frequency.Add(parts[j], 0);
                            cat.frequency[parts[j]] += 1;
                        }
                        cat.total += parts.Length;
                    }
                    cat.id = catExample.Keys.ElementAt(i);
                    textModel.catagories.Add(cat);
                }

                return textModel;
            }

            public static model addTextToModel(string s, string classification, char split, model textModel)
            {
                int i = textModel.catagories.FindIndex((x)=>x.id.Equals(classification));
                string[] parts = s.Split(split);
                if (i == -1) 
                {
                    catagory cat = new catagory();
                    for (int j = 0; j < parts.Length; j++)
                    {
                        if (!cat.frequency.ContainsKey(parts[i])) cat.frequency.Add(parts[i], 0);
                        cat.frequency[parts[i]] += 1;
                    }

                    cat.total = parts.Length;
                    cat.id = classification;
                    textModel.catagories.Add(cat);
                    return textModel;
                } else {
                    catagory cat = textModel.catagories[i];
                    for (int j = 0; j < parts.Length; j++)
                    {
                        if (!cat.frequency.ContainsKey(parts[i])) cat.frequency.Add(parts[i], 0);
                        cat.frequency[parts[i]] += 1;
                    }
                    cat.total += parts.Length;
                    textModel.catagories[i] = cat;
                    return textModel;
                }
            }

            public static string clean(string str, char[] ignore)
            {
                string ret = "";
                for (int i = 0; i < str.Length; i++)
                {
                    char ch = str[i];
                    if (ignore.Contains(ch)) { ret += ch; continue; }
                    if (Char.IsSymbol(ch) || Char.IsSeparator(ch) || Char.IsWhiteSpace(ch) || Char.IsPunctuation(ch)) continue;
                    ret += ch;
                }
                return ret;
            }

            //This will distort data proportions.
            //Not recommended big size, take long time
            public static model trainTextToGoal(model tm, double pg, textTrainDataProvider tdp, bool learnMode, bool fineMode)
            {
                model textModel = tm;
                double percentGoal = pg / 100;
                bool mode = false;
                model prev = textModel;
                double prevScore = 0;

                while(true)
                {
                    Console.WriteLine("[ttg]: Calculating percent score");
                    double score = testTextModel(textModel, tdp);
                    Console.WriteLine("[ttg]: Current percent score: " + score.ToString("0." + new string('#', 339)));
                    if (score.CompareTo(percentGoal) >= 0) break;
                    Console.WriteLine("[ttg]: compare " + score.CompareTo(prevScore));
                    if ((score.CompareTo(prevScore) < 0) && (!mode) && fineMode) 
                    {
                        Console.WriteLine("[ttg]: Switching mode");
                        mode = true;
                        textModel = prev;
                    } 
                    else if ((score.CompareTo(prevScore) < 0) && (mode || !fineMode))
                    {
                        return prev;
                    }

                    prev = textModel;
                    prevScore = score;
                    Console.WriteLine("[ttg]: Starting Pass");
                    for (int i = 0; i < tdp.data.Count; i++)
                    {
                        List<string[]> cat = tdp.getCatagory(i);
                        string expected = tdp.getCatID(i);
                        for (int j = 0; j < cat.Count; j++)
                        {
                            List<object> list = cat[j].ToList<object>();
                            Dictionary<object, double> pred = textModel.predict(list, true);
                            if (same(pred.Values.ToList())) continue;
                            object output = pred.Keys.ElementAt(max(pred));
                            if (!expected.Equals(output))
                            {
                                //Donald Trump: WRONG!
                                //Maybe addfeature and remove feature are swaped because of the word: "the"?
                                for (int k = 0; k < textModel.catagories.Count; k++)
                                {
                                    object catid = textModel.catagories[k].id;
                                    if(output.Equals(catid))
                                    {
                                        //Penalize wrong output

                                        if(mode)
                                        {
                                            for (int n = 0; n < list.Count; n++)
                                            {
                                                List<object> objectTest = new List<object>();
                                                objectTest.Add(list[n]);
                                                if (!expected.Equals(textModel.predict(objectTest, true)))
                                                {
                                                    textModel.removeFeature(catid, list[n], 1);
                                                }
                                                else
                                                {
                                                    textModel.addFeature(expected, list[n], 1);
                                                }
                                            }
                                        } 
                                        else 
                                        {
                                            for (int n = 0; n < list.Count; n++)
                                                textModel.addFeature(catid, list[n], 1);
                                        }

                                    } else if (expected.Equals(catid))
                                    {
                                        //re-enforce correct output

                                        if (mode)
                                        {
                                            for (int n = 0; n < list.Count; n++)
                                            {
                                                List<object> objectTest = new List<object>();
                                                objectTest.Add(list[n]);
                                                if (expected.Equals(textModel.predict(objectTest, true)))
                                                {
                                                    textModel.addFeature(expected, list[n], 1);
                                                }
                                                else
                                                {
                                                    textModel.removeFeature(output, list[n], 1);
                                                }
                                            }
                                        }
                                        else
                                        {
                                            for (int n = 0; n < list.Count; n++)
                                                textModel.removeFeature(catid, list[n], 1);
                                        }
                                    } else if(learnMode) 
                                    {
                                        //Not wrong but not right
                                        //Penalize output
                                        for (int n = 0; n < list.Count; n++)
                                            textModel.removeFeature(catid, list[n], 1);
                                    }
                                }
                            }
                        }
                    }

                    Console.WriteLine("[ttg]: Pass completed");
                }
                return textModel;
            }

            public static double testTextModel(model textModel, textTrainDataProvider tdp)
            {
                List<double> percents = new List<double>();
                int total = 0;
                int correct = 0;
                for (int i = 0; i < tdp.data.Count; i++)
                {
                    List<string[]> cat = tdp.getCatagory(i);
                    string expected = tdp.getCatID(i);
                    for (int j = 0; j < cat.Count; j++)
                    {
                        List<object> list = cat[j].ToList<object>();
                        Dictionary<object, double> pred = textModel.predict(list, true);
                        if (same(pred.Values.ToList())) continue;
                        object output = pred.Keys.ElementAt(max(pred));
                        if (expected.Equals(output)) correct++;
                    }
                    total += cat.Count;
                }
                return (double)correct / (double)total;
            }

            public class textTrainDataProvider
            {
                public Dictionary<string, List<string[]>> data = new Dictionary<string, List<string[]>>();

                public void regCatagory(string id, List<string> rawData, char split)
                {
                    List<string[]> processed = new List<string[]>();
                    foreach (string dataEntry in rawData)
                        processed.Add(dataEntry.Split(split));
                    data.Add(id, processed);
                }

                public List<string[]> getCatagory(int i)
                {
                    return data[data.Keys.ElementAt(i)];
                }

                public string getCatID(int i)
                {
                    return data.Keys.ElementAt(i);
                }
            }

            public static int max(Dictionary<object, double> pred)
            {
                double cm = pred.Values.ElementAt(0);
                int iter = 0;
                for (int i = 1; i < pred.Keys.Count; i++)
                {
                    if (cm < pred.Values.ElementAt(i))
                    {
                        iter = i;
                        cm = pred.Values.ElementAt(i);
                    }
                }

                return iter;
            }

            public static bool same(List<double> pred)
            {
                double last = pred[0];
                for (int i = 1; i < pred.Count; i++)
                {
                    if (!last.Equals(pred[i])) return false;
                }
                return true;
            }
        }
    }
}

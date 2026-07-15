#!/bin/bash

echo ">>> Ожидание доступности HDFS NameNode (192.168.34.2:8020)..."
for i in $(seq 1 60); do
  if bash -c 'echo > /dev/tcp/192.168.34.2/8020' 2>/dev/null; then
    echo ">>> NameNode доступен, продолжаем"
    break
  fi
  echo "Попытка $i/60: порт 8020 пока недоступен, жду 5 секунд..."
  sleep 5
done

echo ">>> Ожидание доступности YARN ResourceManager (192.168.34.2:8032)..."
for i in $(seq 1 60); do
  if bash -c 'echo > /dev/tcp/192.168.34.2/8032' 2>/dev/null; then
    echo ">>> ResourceManager доступен, продолжаем"
    break
  fi
  echo "Попытка $i/60: порт 8032 пока недоступен, жду 5 секунд..."
  sleep 5
done

echo ">>> 1. Создание /createme"
hdfs dfs -mkdir -p /createme

echo ">>> 2. Удаление /delme"
hdfs dfs -rm -r -f /delme

echo ">>> 3. Создание /nonnull.txt"
echo "Some arbitrary content for the task." | hdfs dfs -put -f - /nonnull.txt

echo ">>> 4. WordCount для /shadow.txt через YARN"
hdfs dfs -rm -r -f /output_wordcount
hadoop jar "$HADOOP_HOME/share/hadoop/mapreduce/hadoop-mapreduce-examples-${HADOOP_VERSION}.jar" \
  wordcount /shadow.txt /output_wordcount

echo ">>> 5. Подсчёт точных вхождений 'Innsmouth'"
COUNT=$(hdfs dfs -cat /output_wordcount/part-r-00000 | awk -F'\t' '$1=="Innsmouth"{print $2}')
if [ -z "$COUNT" ]; then
  COUNT=0
fi
echo "$COUNT" | hdfs dfs -put -f - /whataboutinsmouth.txt

echo ">>> Готово. Innsmouth встречается $COUNT раз(а)"
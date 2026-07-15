FROM eclipse-temurin:8-jdk

ENV HADOOP_VERSION=3.3.6
ENV HADOOP_HOME=/opt/hadoop
ENV HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
ENV PATH=$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$PATH

COPY hadoop-${HADOOP_VERSION}.tar.gz /tmp/hadoop.tar.gz
RUN tar -xzf /tmp/hadoop.tar.gz -C /opt && \
    mv /opt/hadoop-${HADOOP_VERSION} $HADOOP_HOME && \
    rm /tmp/hadoop.tar.gz

COPY core-site.xml   $HADOOP_CONF_DIR/core-site.xml
COPY yarn-site.xml   $HADOOP_CONF_DIR/yarn-site.xml
COPY mapred-site.xml $HADOOP_CONF_DIR/mapred-site.xml
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# RUN echo "192.168.34.2 tank-rxS7MJsm5U.cr-int" >> /etc/hosts

ENTRYPOINT ["/entrypoint.sh"]
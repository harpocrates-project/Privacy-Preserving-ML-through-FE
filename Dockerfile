# Build Stage
FROM golang:1.23 AS build

WORKDIR /SPADE
COPY . ./
RUN go build -o /opt/server ./usecases/hypnogram/cmd/server/
RUN go build -o /opt/analyst ./usecases/hypnogram/cmd/analyst/
RUN go build -o /opt/user ./usecases/hypnogram/cmd/user/

FROM python:3-slim AS runtime

COPY --from=build /opt/server /opt/server
COPY --from=build /opt/analyst /opt/analyst
COPY --from=build /opt/user /opt/user
COPY --from=build /SPADE/usecases/hypnogram/gui/utils/plot_hypnogram.py /opt/plot_hypnogram.py

RUN pip install --no-cache-dir matplotlib

RUN echo

COPY <<"EOT" /opt/analyst_command.sh
#!/bin/bash
HOST=$1
SCAN=$2
QUERY=$3
/opt/analyst ${HOST} ${SCAN} ${QUERY} /output/${SCAN}_${QUERY}.txt && python /opt/plot_hypnogram.py /output/${SCAN}_${QUERY}.txt --plot_file /output/${SCAN}_${QUERY}.jpg
EOT

RUN chmod +x /opt/analyst_command.sh

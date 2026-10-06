# ZarrSwarm node. Data port 7881 (publish it on public nodes); control API stays inside the container.
#   docker build -t zarrswarm .
#   docker run -d --name zt -v zt-home:/zt -v /data:/data:ro -p 7881:7881 zarrswarm node
#   docker exec zt zt init --join ztnet://ID@HOST:7881   (once, then restart), docker exec zt zt status
FROM python:3.12-slim
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY zarrswarm ./zarrswarm
COPY zarr_torrent ./zarr_torrent
RUN pip install --no-cache-dir ".[netcdf]"
ENV ZT_HOME=/zt
VOLUME /zt
EXPOSE 7881
ENTRYPOINT ["zt"]
CMD ["node"]

FROM nginx:1.27-alpine

WORKDIR /usr/share/nginx/html

# Clean the default Nginx static assets.
RUN rm -rf ./*

# Copy static frontend files.
COPY index.html ./
COPY interface_API.html ./
COPY css/ ./css/
COPY fontes/ ./fontes/
COPY imgs/ ./imgs/

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
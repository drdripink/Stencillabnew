FROM nginx:1.25-alpine
RUN rm -rf /usr/share/nginx/html/* \
 && rm -f /etc/nginx/conf.d/default.conf
COPY index.html /usr/share/nginx/html/index.html
RUN printf 'server {\n listen 8080;\n server_name _;\n root /usr/share/nginx/html;\n index index.html;\n location / {\n  try_files $uri /index.html =404;\n }\n}\n' > /etc/nginx/conf.d/app.conf
RUN nginx -t && ls -la /usr/share/nginx/html/
EXPOSE 8080
CMD ["nginx", "-g", "daemon off;"]

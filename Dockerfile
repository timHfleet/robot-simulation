# Stage 1: build the React front end
FROM node:20-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# Stage 2: serve the static site with nginx on $PORT (Render sets it).
FROM nginx:1.27-alpine
COPY --from=web /web/dist /usr/share/nginx/html
# The nginx image fills in ${PORT} from the environment at startup.
COPY deploy/default.conf.template /etc/nginx/templates/default.conf.template
ENV PORT=8000
EXPOSE 8000

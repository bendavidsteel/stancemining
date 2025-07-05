# Build Stage
FROM node:18-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .

ARG API_BASE_URL

ENV REACT_APP_API_BASE_URL=$API_BASE_URL

RUN npm run build
 
# Production Stage
FROM nginx:stable-alpine AS production
COPY --from=build /app/build /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
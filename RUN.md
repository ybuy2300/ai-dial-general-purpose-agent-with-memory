## Running the app


### Down previous docker containers if any
```
$ docker compose down
```

### Clear the old data
```
$ rm -rf core-data/ core-logs/
```

### Build and up docker compose in detached mode
```
$ docker-compose up --build -d 
```

### Run the app
```
$ python -m task.app
```

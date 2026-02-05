build:
	podman build --build-arg UID=$$(id -u) --build-arg GID=$$(id -g) -t lemonslice .

run:
	find ./src -type d -name __pycache__ -exec rm -rf {} +
	podman run -e HOME=/app --name lemonslice -it -p 7880:7880 --user root -v ./src:/app/src:Z lemonslice /bin/bash
	

attach:
	podman exec -it lemonslice /bin/bash

remove:
	podman rm -f lemonslice
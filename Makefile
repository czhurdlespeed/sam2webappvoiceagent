build:
	podman build --build-arg UID=$$(id -u) --build-arg GID=$$(id -g) -t lemonslice .

run:
	podman run --name lemonslice -it -p 7880:7880 -v ./src:/app/src:z lemonslice /bin/bash

attach:
	podman exec -it lemonslice /bin/bash

remove:
	podman rm -f lemonslice
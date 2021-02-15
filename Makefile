SRC=src/*.cc

compile:
	g++ -std=c++17 -Wall -Wextra -Werror -O5 -o cmkv ${SRC} -lfreeimage

clean:
	${RM} cmkv

.PHONY: compile clean

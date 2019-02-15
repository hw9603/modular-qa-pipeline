FILENAME == ARGV[1] { answer[FNR] = $1}
FILENAME == ARGV[1] { one[FNR]=$2 }
FILENAME == ARGV[2] { two[FNR]=$2 }
FILENAME == ARGV[3] { three[FNR]=$2 }
FILENAME == ARGV[4] { four[FNR]=$2 }
FILENAME == ARGV[5] { five[FNR]=$2 }
FILENAME == ARGV[6] { six[FNR]=$2 }

END {
    for (i=1; i<=length(one); i++) {
        print answer[i], ",", one[i], ",", two[i], ",", three[i], ",", four[i], ",", five[i], ",", six[i]
    }
}
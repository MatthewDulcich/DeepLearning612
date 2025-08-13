#!/bin/bash
#
# Helper script to unpack conda-pack tarball

# Globals
STATUSFILE="EXTRACTED"
TIMEOUT=300 #Seconds
# time between polling for STATUSFILE
TIMESTEP=5 #Seconds


usage()
{	cat - <<EOF >&2
$0: Helper script to unpack conda-test tarball

Usage:
$0 [ -v ] \\
	[ -d DESTDIR ] \\
	-f TARBALL \\
	[ -s STATUSFILE ] \\
	[ -t TIMEOUT ] 

This script will extract the tarball specified
by TARBALL into the directory given by
DESTDIR if DESTDIR does not already exist.
After extracting the TARBALL, it will add an 
empty file named STATUSFILE to DESTDIR.

STATUSFILE defaults to $STATUSFILE.
DESTDIR defaults to "/tmp/\$USER/BASENAME"
where BASENAME is the basename of TARBALL
(i.e. remove directory and extension(s) from
TARBALL).

The tar extraction is done with the --auto-compress
flag, so standard extensions will be recognized
and decompressed accordingly.

If DESTDIR exists, we do *not* extract the
tarball, but instead wait (up to TIMEOUT
seconds) for the STATUSFILE file to appear
in DESTDIR.  If the file appears before
the TIMEOUT is exceeded, the script will 
return successfully.

TIMEOUT defaults to $TIMEOUT seconds

If the -v flag is given, be more verbose
about what the script is doing.

Exit codes:
0 on success (either tarball was
successfully extracted, or the DESTDIR/STATUSFILE
was found before TIMEOUT reached)
1 if there were errors extracting the tarball
2 if DESTDIR was found but STATUSFILE
was not found before the TIMEOUT was reached.
3 for other fatal errors
EOF
}

#-----------------------------------------------
#		Main
#-----------------------------------------------

#--------- Process arguments
DESTDIR=
TARBALL=
VERBOSE="no"

while getopts :hd:f:s:t:v arg
do
        case $arg in
        h)
                usage
		exit 0
                ;;

	d)	DESTDIR="$OPTARG"
		;;

	f)	TARBALL="$OPTARG"
		;;

	s)	STATUSFILE="$OPTARG"
		;;

	t)	TIMEOUT="$OPTARG"
		;;

	v)	VERBOSE="yes"
		;;

        *) 
		usage
		echo "Illegal option -$OPTARG, aborting." >&2
		exit 1
	esac
done
#COUNT=`echo $OPTIND -1 | bc`
COUNT=$(( $OPTIND - 1 ))
shift $COUNT

if [ $# -gt 0 ]; then
	usage
	echo >&2 "[ERROR] Unrecognized options $@, aborting"
	exit 3
fi

# Ensure required arguments were given

if [ "x$TARBALL" == "x" ]; then
	usage
	echo >&2 "[ERROR] Missing required parameter -f, aborting"
	exit 3
fi

# Default DESTDIR if needed
if [ "x$DESTDIR" == "x" ]; then
	# strip directory and all extensions from TARBALL
	DESTDIR=$( basename $TARBALL | sed -e 's/\..*$//' )
	if [ "x$DESTDIR" = "x" ]; then
		echo >&2 "[ERROR] DESTDIR not given and cannot default"
		exit 3
	fi
	DESTDIR="/tmp/${USER}/${DESTDIR}"
fi

# Validate TIMEOUT and STATUSFILE
if [ $TIMEOUT -lt 0 ]; then
	usage
	echo >&2 "[ERROR] Timeout must be greater than or equal to  0"
	exit 3
fi

if [ "x$STATUSFILE" == "x" ]; then
	usage
	echo >&2 "[ERROR] Illegal value '$STATUSFILE' for STATUSFILE"
	exit 3
fi

if [ "x$VERBOSE" == "xyes" ]; then
	echo >&2 "[VERBOSE] TARBALL=$TARBALL";
	echo >&2 "[VERBOSE] DESTDIR=$DESTDIR";
	echo >&2 "[VERBOSE] STATUSFILE=$STATUSFILE";
	echo >&2 "[VERBOSE] TIMEOUT=$TIMEOUT";
fi

#--------- Do the actual work

if [ -d $DESTDIR ]; then
	# DESTDIR directory found
	WAITED=0

	if [ "x$VERBOSE" == "xyes" ]; then
		echo >&2 "[VERBOSE] $DESTDIR found, waiting for $STATUSFILE"
	fi

	while /bin/true
	do
		if [ -e "$DESTDIR/$STATUSFILE" ]; then
			if [ "x$VERBOSE" == "xyes" ]; then
				echo >&2 "[VERBOSE] $STATUSFILE found, successful"
			fi
			exit 0
		fi

		sleep $TIMESTEP
		WAITED=$(( WAITED + TIMESTEP ))
		if [ $WAITED -ge $TIMEOUT ]; then
			echo >&2 "[ERROR] STATUSFILE $STATUSFILE not found after " \
				"$WAITED seconds, timing out"
			exit 2
		fi
	done

	echo >&2 "[ERROR] Should not reach here A"
	exit 3
else
	# DESTDIR not found

	if [ "x$VERBOSE" == "xyes" ]; then
		echo >&2 "[VERBOSE] $DESTDIR not found, untarring $TARBALL..."
	fi

	# Create directory
	mkdirhier $DESTDIR
	if [ $? -ne 0 ]; then
		echo >&2 "[ERROR] Error mkdirhier $DESTDIR, aborting"
		exit 1
	fi

	# untar tarball
	tar -xf $TARBALL --auto-compress -C $DESTDIR
	if [ $? -ne 0 ]; then
		echo >&2 "[ERROR] Error untarring $TARBALL, aborting"
		exit 1
	fi

	# Touch STATUSFILE
	touch "${DESTDIR}/${STATUSFILE}"
	if [ $? -ne 0 ]; then
		echo >&2 "[ERROR] Error touching $STATUSFILE, aborting"
		exit 1
	fi

	# Success
	if [ "x$VERBOSE" == "xyes" ]; then
		echo >&2 "[VERBOSE] Successfully untarred $TARBALL..."
	fi
	exit 0
fi

echo >&2 "[ERROR] Should not reach here B"
exit 3


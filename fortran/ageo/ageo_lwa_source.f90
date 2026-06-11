        program main

        use NETCDF

!   ageostrophic LWA source S_q = LWA-projection of  - nabla.(va.zeta) - beta.va
!   Climatology version: takes (year, month) as command-line args.
!   Saves a single monthly NetCDF with  AOUT_baro (column-integrated, density-
!   weighted)  and LWA_baro (= astar1+astar2 column avg) for the barotropic
!   LWA budget closure.  Optionally also saves FORCE_baro (raw force column avg).
!   Input/output directories come from command-line args 3 and 4 or from the
!   AGEO_INDIR and AGEO_OUTDIR environment variables (no internal defaults).
!   Compile: gfortran -O2 ageo_lwa_source.f90 -I$NETCDF_INC -L$NETCDF_LIB \
!            -lnetcdff -lnetcdf -o ageo_lwa_source.out
!   Run:    ./ageo_lwa_source.out 2014 11 /path/to/qg_binaries /path/to/out

        integer,parameter :: imax = 360, JMAX = 181, KMAX = 97
        integer,parameter :: nd = 91, nnd = 181, jd = 86
        common /array/ pv(imax,jmax,kmax)
        common /brray/ uu(imax,jmax,kmax)
        common /bbray/ vv(imax,jmax,kmax)
        common /beray/ avort(imax,jmax,kmax)
        common /bdray/ zz(imax,jmax,kmax)
        common /bcray/ pt(imax,jmax,kmax)
        common /beray/ ug(imax,jmax,kmax)
        common /bfray/ vg(imax,jmax,kmax)
        common /bgray/ ua(imax,jmax,kmax)
        common /bghay/ va(imax,jmax,kmax)
        common /cghay/ force(imax,jmax,kmax)
        common /bdray/ stats(kmax),statn(kmax),ts0(kmax),tn0(kmax)
        common /crray/ tb(kmax),tg(kmax),ug0(kmax),ug1(kmax)
        common /erray/ astar1(imax,nd,kmax)
        common /errayy/ astar2(imax,nd,kmax)
        common /eerayx/ ua0(imax,nd,kmax)
        common /frrbyx/ aout(imax,nd,kmax)
        common /irray/ qref(91,kmax),u(91,kmax)
        common /jrray/ qbar(91,kmax),ubar(91,kmax),tbar(91,kmax)
        common /krray/ uref(jd,kmax),tref(jd,kmax),fawa(91,kmax)

!       barotropic accumulators (per snapshot t -> mm-loop step)
!       all on (imax, nd, time) NH grid: j=1..91 = lat 0..90 N
        real, allocatable :: aout_baro(:,:,:)
        real, allocatable :: lwa_baro(:,:,:)
        real, allocatable :: force_baro(:,:,:)

        integer :: md(12)
        real :: z(kmax)

        character(len=512) :: fn,fu,ft,fv,fz,fvt,fx,outpath
        character(len=512) :: indir, outdir
        character(len=4)   :: fy
        character(len=4)   :: fn2(12)
        character(len=512) :: arg

        integer :: ncid, status, vid_aout, vid_lwa, vid_force
        integer :: dim_lon, dim_lat, dim_time
        integer :: m, n, mm, ntime, k, j, i, jj, istat
        real    :: a, pi, om, dp, dz, h, r, rkappa, dc
        real    :: phi, phi0, phip, phim, phi1, fff, fff0, fffp, fffm
        real    :: cor, beta, ab, aa, zk

        a = 6378000.
        pi = acos(-1.)
        om = 7.29e-5
        dp = pi/180.
        dz = 500.
        h = 7000.
        r = 287.
        rkappa = r/1004.

        do k = 1, kmax
           z(k) = dz*float(k-1)
        enddo

        if (command_argument_count() .lt. 2) then
           write(0,*) 'usage: ageo_lwa_source.out <year> <month>', &
                      ' [<input_dir> <output_dir>]'
           write(0,*) ' input_dir/output_dir may instead be supplied via'
           write(0,*) ' the AGEO_INDIR and AGEO_OUTDIR environment variables.'
           stop 1
        endif
        call get_command_argument(1, arg); read(arg,*) m
        call get_command_argument(2, arg); read(arg,*) n
        write(fy,'(i4)') m
        if (command_argument_count() .ge. 4) then
           call get_command_argument(3, arg); indir  = trim(arg)
           call get_command_argument(4, arg); outdir = trim(arg)
        else
           call get_environment_variable("AGEO_INDIR", indir, status=istat)
           if (istat .ne. 0 .or. len_trim(indir) .eq. 0) then
              write(0,*) 'error: input dir not set (arg 3 or AGEO_INDIR)'
              stop 1
           endif
           call get_environment_variable("AGEO_OUTDIR", outdir, status=istat)
           if (istat .ne. 0 .or. len_trim(outdir) .eq. 0) then
              write(0,*) 'error: output dir not set (arg 4 or AGEO_OUTDIR)'
              stop 1
           endif
        endif

        md = (/31,28,31,30,31,30,31,31,30,31,30,31/)
        if (mod(m,4).eq.0 .and. (mod(m,100).ne.0 .or. mod(m,400).eq.0)) md(2) = 29
        ntime = md(n)*4
        allocate(aout_baro(imax,nd,ntime))
        allocate(lwa_baro(imax,nd,ntime))
        allocate(force_baro(imax,nd,ntime))
        aout_baro = 0.; lwa_baro = 0.; force_baro = 0.

        fn2(1)  = '_01_'; fn2(2)  = '_02_'; fn2(3)  = '_03_'
        fn2(4)  = '_04_'; fn2(5)  = '_05_'; fn2(6)  = '_06_'
        fn2(7)  = '_07_'; fn2(8)  = '_08_'; fn2(9)  = '_09_'
        fn2(10) = '_10_'; fn2(11) = '_11_'; fn2(12) = '_12_'

        fn  = trim(indir)//'/'//fy//fn2(n)//'QGPV'
        fu  = trim(indir)//'/'//fy//fn2(n)//'QGU'
        ft  = trim(indir)//'/'//fy//fn2(n)//'QGT'
        fv  = trim(indir)//'/'//fy//fn2(n)//'QGV'
        fz  = trim(indir)//'/'//fy//fn2(n)//'QGZ'
        fvt = trim(indir)//'/'//fy//fn2(n)//'QVORT'
        fx  = trim(indir)//'/'//fy//fn2(n)//'QGREF_N'
        outpath = trim(outdir)//'/'//fy//fn2(n)//'AOUTbaro_N.nc'

        write(6,*) 'processing ',fy,n,' ntime=',ntime
        write(6,*) 'output     ',trim(outpath)

        open(34,file=fz,  form='unformatted',status='old')
        open(35,file=fn,  form='unformatted',status='old')
        open(36,file=fu,  form='unformatted',status='old')
        open(37,file=ft,  form='unformatted',status='old')
        open(39,file=fv,  form='unformatted',status='old')
        open(51,file=fvt, form='unformatted',status='old')
        open(40,file=fx,  form='unformatted',status='old')

        dc = dz/6745.348

        do mm = 1, ntime

           read(35) pv
           read(36) uu
           read(34) zz
           read(39) vv
           read(51) avort
           read(37) pt,tn0,ts0,statn,stats
           read(40) qref,uref,tref,fawa,ubar,tbar

           tg(:) = tn0(:)

! **** geostrophic wind ****
           do j = 1, 86
              phi = dp*float(j-1) - 0.5*pi
              fff = 2.*om*sin(phi)
              do i = 2, imax-1
                 vg(i,j,:) = (zz(i+1,j,:)-zz(i-1,j,:))/(2.*a*fff*cos(phi)*dp)
              enddo
              vg(1,j,:)    = (zz(2,j,:)-zz(imax,j,:))/(2.*a*fff*cos(phi)*dp)
              vg(imax,j,:) = (zz(1,j,:)-zz(imax-1,j,:))/(2.*a*fff*cos(phi)*dp)
           enddo
           do j = 96, jmax
              phi = dp*float(j-1) - 0.5*pi
              fff = 2.*om*sin(phi)
              do i = 2, imax-1
                 vg(i,j,:) = (zz(i+1,j,:)-zz(i-1,j,:))/(2.*a*fff*cos(phi)*dp)
              enddo
              vg(1,j,:)    = (zz(2,j,:)-zz(imax,j,:))/(2.*a*fff*cos(phi)*dp)
              vg(imax,j,:) = (zz(1,j,:)-zz(imax-1,j,:))/(2.*a*fff*cos(phi)*dp)
           enddo

           do j = 2, 86
              phi0 = dp*float(j-1) - 0.5*pi
              fff = 2.*om*sin(phi0)
              ug(:,j,:) = -(zz(:,j+1,:)-zz(:,j-1,:))/(2.*a*fff*dp)
           enddo
           ug0(:) = 0.
           do i = 1, imax
              ug0(:) = ug0(:) + ug(i,2,:)/float(imax)
           enddo
           do i = 1, imax
              ug(i,1,:) = ug0(:)
           enddo
           do j = 96, jmax-1
              phi0 = dp*float(j-1) - 0.5*pi
              fff = 2.*om*sin(phi0)
              ug(:,j,:) = -(zz(:,j+1,:)-zz(:,j-1,:))/(2.*a*fff*dp)
           enddo
           ug1(:) = 0.
           do i = 1, imax
              ug1(:) = ug1(:) + ug(i,jmax-1,:)/float(imax)
           enddo
           do i = 1, imax
              ug(i,jmax,:) = ug1(:)
           enddo

! **** ageostrophic wind ****
           do j = 1, 86
              va(:,j,:) = vv(:,j,:) - vg(:,j,:)
              ua(:,j,:) = uu(:,j,:) - ug(:,j,:)
           enddo
           do j = 96, jmax
              va(:,j,:) = vv(:,j,:) - vg(:,j,:)
              ua(:,j,:) = uu(:,j,:) - ug(:,j,:)
           enddo

! **** ageostrophic forcing  -beta*va - nabla_h.(va*zeta)  ****
           do j = 2, 86
              phi0 = dp*float(j-1) - 0.5*pi
              phip = dp*float(j)   - 0.5*pi
              phim = dp*float(j-2) - 0.5*pi
              beta = 2.*om*cos(phi0)/a
              fff0 = 2.*om*sin(phi0)
              fffp = 2.*om*sin(phip)
              fffm = 2.*om*sin(phim)
              force(:,j,:) = -beta*va(:,j,:)
              force(:,j,:) = force(:,j,:) -                                 &
                 (va(:,j+1,:)*(avort(:,j+1,:)-fffp)*cos(phip)               &
                 -va(:,j-1,:)*(avort(:,j-1,:)-fffm)*cos(phim))              &
                 /(2.*a*cos(phi0)*dp)
              do i = 2, imax-1
                 force(i,j,:) = force(i,j,:) -                              &
                    (ua(i+1,j,:)*(avort(i+1,j,:)-fff0)                      &
                    -ua(i-1,j,:)*(avort(i-1,j,:)-fff0))                     &
                    /(2.*a*cos(phi0)*dp)
              enddo
              force(1,j,:)    = force(1,j,:) -                              &
                 (ua(2,j,:)*(avort(2,j,:)-fff0)                             &
                 -ua(imax,j,:)*(avort(imax,j,:)-fff0))                      &
                 /(2.*a*cos(phi0)*dp)
              force(imax,j,:) = force(imax,j,:) -                           &
                 (ua(1,j,:)*(avort(1,j,:)-fff0)                             &
                 -ua(imax-1,j,:)*(avort(imax-1,j,:)-fff0))                  &
                 /(2.*a*cos(phi0)*dp)
           enddo
           do j = 96, jmax-1
              phi0 = dp*float(j-1) - 0.5*pi
              phip = dp*float(j)   - 0.5*pi
              phim = dp*float(j-2) - 0.5*pi
              beta = 2.*om*cos(phi0)/a
              fff0 = 2.*om*sin(phi0)
              fffp = 2.*om*sin(phip)
              fffm = 2.*om*sin(phim)
              force(:,j,:) = -beta*va(:,j,:)
              force(:,j,:) = force(:,j,:) -                                 &
                 (va(:,j+1,:)*(avort(:,j+1,:)-fffp)*cos(phip)               &
                 -va(:,j-1,:)*(avort(:,j-1,:)-fffm)*cos(phim))              &
                 /(2.*a*cos(phi0)*dp)
              do i = 2, imax-1
                 force(i,j,:) = force(i,j,:) -                              &
                    (ua(i+1,j,:)*(avort(i+1,j,:)-fff0)                      &
                    -ua(i-1,j,:)*(avort(i-1,j,:)-fff0))                     &
                    /(2.*a*cos(phi0)*dp)
              enddo
              force(1,j,:)    = force(1,j,:) -                              &
                 (ua(2,j,:)*(avort(2,j,:)-fff0)                             &
                 -ua(imax,j,:)*(avort(imax,j,:)-fff0))                      &
                 /(2.*a*cos(phi0)*dp)
              force(imax,j,:) = force(imax,j,:) -                           &
                 (ua(1,j,:)*(avort(1,j,:)-fff0)                             &
                 -ua(imax-1,j,:)*(avort(imax-1,j,:)-fff0))                  &
                 /(2.*a*cos(phi0)*dp)
           enddo

! **** wave activity (LWA) and LWA-projected ageostrophic source AOUT ****
!     k=11..95 -> z=5..47 km.  Lower bound 5 km drops sub-mountaintop levels
!     where QGZ has spurious values (especially in MPAS over Tibet/Pamir).
           do k = 11, 95
              zk = dz*float(k-1)
              do i = 1, imax
                 do j = 6, nd-1
                    astar1(i,j,k) = 0.
                    astar2(i,j,k) = 0.
                    aout(i,j,k)   = 0.
                    phi0 = dp*float(j-1)
                    do jj = 1, nd
                       phi1 = dp*float(jj-1)
                       aa = a*dp*cos(phi1)
                       if ((pv(i,jj+90,k)-qref(j,k)).le.0. .and. jj.ge.j) then
                          astar2(i,j,k) = astar2(i,j,k)                     &
                                  - (pv(i,jj+90,k)-qref(j,k))*aa
                          aout(i,j,k)   = aout(i,j,k) + force(i,jj+90,k)*aa
                       endif
                       if ((pv(i,jj+90,k)-qref(j,k)).gt.0. .and. jj.lt.j) then
                          astar1(i,j,k) = astar1(i,j,k)                     &
                                  + (pv(i,jj+90,k)-qref(j,k))*aa
                          aout(i,j,k)   = aout(i,j,k) - force(i,jj+90,k)*aa
                       endif
                    enddo

! barotropic (density-weighted column) accumulation
                    aout_baro(i,j,mm) = aout_baro(i,j,mm)                   &
                                + aout(i,j,k)  *exp(-zk/h)*dc
                    lwa_baro(i,j,mm)  = lwa_baro(i,j,mm)                    &
                                + (astar1(i,j,k)+astar2(i,j,k))             &
                                  *exp(-zk/h)*dc
                 enddo
              enddo
           enddo

! barotropic raw force (NH only, nd=91: j=1..91 -> lat 0..90 N)
!     same vertical range as aout_baro/lwa_baro (k=11..95, z=5..47 km).
           do k = 11, 95
              zk = dz*float(k-1)
              do j = 1, nd
                 force_baro(:,j,mm) = force_baro(:,j,mm)                    &
                                  + force(:,j+90,k)*exp(-zk/h)*dc
              enddo
           enddo

           write(6,*) fy,n,mm

        enddo

        close(34); close(35); close(36); close(37)
        close(39); close(40); close(51)

! **** write NetCDF ****
        status = nf90_create(trim(outpath), nf90_clobber, ncid)
        status = nf90_def_dim(ncid, 'longitude', imax,  dim_lon)
        status = nf90_def_dim(ncid, 'latitude',  nd,    dim_lat)
        status = nf90_def_dim(ncid, 'time',      ntime, dim_time)
        status = nf90_def_var(ncid, 'aout_baro', nf90_float,                 &
                              (/dim_lon, dim_lat, dim_time/), vid_aout)
        status = nf90_put_att(ncid, vid_aout, 'long_name',                   &
                              'LWA-projected ageostrophic forcing, density-weighted column average')
        status = nf90_put_att(ncid, vid_aout, 'units', 'm s-2')
        status = nf90_def_var(ncid, 'lwa_baro',  nf90_float,                 &
                              (/dim_lon, dim_lat, dim_time/), vid_lwa)
        status = nf90_put_att(ncid, vid_lwa, 'long_name',                    &
                              'LWA cos(phi), density-weighted column average')
        status = nf90_put_att(ncid, vid_lwa, 'units', 'm s-1')
        status = nf90_def_var(ncid, 'force_baro', nf90_float,                &
                              (/dim_lon, dim_lat, dim_time/), vid_force)
        status = nf90_put_att(ncid, vid_force, 'long_name',                  &
                              'raw -beta va - div_h(va*zeta), density-weighted column average')
        status = nf90_put_att(ncid, vid_force, 'units', 's-2')
        status = nf90_put_att(ncid, NF90_GLOBAL, 'source',                   &
                              'ageo_lwa_source.f90 (downstream-et-lwa)')
        status = nf90_put_att(ncid, NF90_GLOBAL, 'year',  m)
        status = nf90_put_att(ncid, NF90_GLOBAL, 'month', n)
        status = nf90_put_att(ncid, NF90_GLOBAL, 'dt_hours', 6)
        status = nf90_put_att(ncid, NF90_GLOBAL, 'vert_k_range', '11..95')
        status = nf90_put_att(ncid, NF90_GLOBAL, 'vert_z_range_km', '5..47')
        status = nf90_enddef(ncid)
        status = nf90_put_var(ncid, vid_aout,  aout_baro)
        status = nf90_put_var(ncid, vid_lwa,   lwa_baro)
        status = nf90_put_var(ncid, vid_force, force_baro)
        status = nf90_close(ncid)

        write(6,*) 'wrote ', trim(outpath)

        deallocate(aout_baro, lwa_baro, force_baro)

        stop
        end program main

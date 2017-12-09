!=============================================================================================
!
! This Kernel Represents the GPP Self-Energy Summations in BerkeleyGW
!
! Run like:
!
!  gppkernel.x <number_bands> <number_valence_bands> <number_plane_waves> <nodes_per_mpi_group> <gppsum>
!
! Example run:
!
!  gppkernel.x 8 2 10000 20
!
!==============================================================================================


program gppkernel

      implicit none

      integer :: NTHREADS,TID,OMP_GET_NUM_THREADS,OMP_GET_THREAD_NUM
      integer :: n1
      integer :: ispin, iw
      integer :: number_bands,nvband,ncouls,ngpown
      integer :: nodes_per_group
      integer :: nstart, nend
      integer :: my_igp, indigp, ig, igp, igmax,igbeg,igend
      integer,parameter:: igblk=512
      !integer,parameter:: igblk=512000
      integer, allocatable :: indinv(:), inv_igp_index(:)
      complex(kind((1.0d0,1.0d0))) :: achstemp,achxtemp,matngmatmgp,matngpmatmg,wdiff,mygpvar1,mygpvar2,schstemp
      complex(kind((1.0d0,1.0d0))) ::schs,wx_array(3),sch,ssx,ssxt,scht
      complex(kind((1.0d0,1.0d0))) :: wtilde,halfinvwtilde,wtilde2,Omega2
      complex(kind((1.0d0,1.0d0))), allocatable :: aqsmtemp(:,:), aqsntemp(:,:), I_eps_array(:,:), acht_n1_loc(:)
      complex(kind((1.0d0,1.0d0))), allocatable :: asxtemp(:),achtemp(:),ssx_array(:),sch_array(:),ssxa(:),scha(:),wtilde_array(:,:)
      complex(kind((1.0d0,1.0d0))) :: delw, cden
      complex(kind((1.0d0,1.0d0))) :: scha_temp
      real(kind(1.0d0)) :: scha_mult
      real(kind(1.0d0)) :: e_n1kq, e_lk, dw, delwr, wdiffr
      real(kind(1.0d0)) :: tol, gamma, sexcut
      real(kind(1.0d0)) :: wx,occ,occfact,tempval
      real(kind(1.0d0)) :: wxt,rden,ssxcutoff,delw2
      real(kind(1.0d0)), allocatable :: ekq(:,:), vcoul(:)
      real(kind(1.0d0)) :: limitone,limittwo
      real(kind(1.0d0)) :: starttime, endtime
      real(kind(1.0d0)) :: starttime_stat, endtime_stat
      real(kind(1.0d0)) :: starttime_noFlagOCC, endtime_noFlagOCC, totaltime_noFlagOCC
      real(kind(1.0d0)) :: time_stat
      real(kind(1.0d0)) :: starttime_dyn, endtime_dyn
      real(kind(1.0d0)) :: time_dyn
      logical :: flag_occ
      CHARACTER(len=32) :: arg
      integer npes,mype,ierr
!      include 'mpif.h'

!      call mpi_init(ierr)

      ! Initialize MPI on the node. npes is number of ranks per node
!      call mpi_comm_size(mpi_comm_world,npes,ierr)
!      call mpi_comm_rank(mpi_comm_world,mype,ierr)

      time_stat = 0D0
      time_dyn = 0D0
      starttime_noFlagOCC = 0D0
      endtime_noFlagOCC = 0D0
      totaltime_noFlagOCC = 0D0

!$OMP PARALLEL PRIVATE(NTHREADS, TID)
      TID = OMP_GET_THREAD_NUM()
      IF (TID .EQ. 0 .and. mype==0) THEN
        NTHREADS = OMP_GET_NUM_THREADS()
        PRINT *, 'Number of threads = ', NTHREADS
        write(6,*) "Number of ranks per node = ",npes
      END IF
!$OMP END PARALLEL

! We start off in the body of loop over the various tasks. Each MPI task has communicated data it owns to everyone

! These should be read in
!      if(mype == 0)then
        CALL getarg(1, arg)
        READ(arg,*) number_bands
        CALL getarg(2, arg)
        READ(arg,*) nvband
        CALL getarg(3, arg)
        READ(arg,*) ncouls
        CALL getarg(4, arg)
        READ(arg,*) nodes_per_group
        CALL getarg(5, arg)
!      endif

!      call mpi_bcast(number_bands,1,mpi_integer,0,mpi_comm_world,ierr)
!      call mpi_bcast(nvband,      1,mpi_integer,0,mpi_comm_world,ierr)
!      call mpi_bcast(ncouls,      1,mpi_integer,0,mpi_comm_world,ierr)
!      call mpi_bcast(ngpown,      1,mpi_integer,0,mpi_comm_world,ierr)
!      call mpi_bcast(nodes_per_group,      1,mpi_integer,0,mpi_comm_world,ierr)

      ! ngpown is number of gvectors per mpi task
      npes=1 !Rahul - 8 ranks per node
      ngpown = ncouls / ( nodes_per_group * npes )


      e_lk = 10D0
      dw = 1D0
      nstart = 1
      nend = 3

!      if(mype==1)then
        write(6,*) "number_bands = ",number_bands
        write(6,*) "nvband = ",nvband
        write(6,*) "ncouls = igmax = ",ncouls
        write(6,*) "ngpown = ",ngpown
        write(6,*) "nend-nstart = ",nend-nstart
!      endif

      ALLOCATE(vcoul(ncouls))
      vcoul = 1D0

      ALLOCATE(asxtemp(nend-nstart+1))
      asxtemp = 0D0

      ALLOCATE(achtemp(nend-nstart+1))
      achtemp = 0D0

      ALLOCATE(ekq(number_bands,1))
      ekq = 6D0

      ALLOCATE(wtilde_array(ncouls,ngpown))
      wtilde_array = (0.5d0,0.5d0)
      if(mype==0)write(6,*)"Size of wtilde_array = ",(ncouls*ngpown*2.0*8)/(1024**2)," Mbytes"

      ALLOCATE(aqsntemp(ncouls,number_bands))
      if(mype==0)write(6,*)"Size of aqsntemp = ",(ncouls*number_bands*2.0*8)/1024**2," Mbytes"
      ALLOCATE(aqsmtemp(ncouls,number_bands))
      aqsmtemp = (0.5D0,0.5D0)
      aqsntemp = (0.5D0,0.5D0)

      ALLOCATE(I_eps_array(ncouls,ngpown))
      if(mype==0)write(6,*)"Size of I_eps_array = ",(ncouls*ngpown*2.0*8)/1024**2," Mbytes"
      I_eps_array = (0.5D0,0.5D0)

      ALLOCATE(inv_igp_index(ngpown))
      ALLOCATE(indinv(ncouls))

      ALLOCATE(acht_n1_loc(number_bands))

!       do ig = 1, ngpown
!         inv_igp_index(ig) = ig
!       enddo
      !  assumes a better mpi work distribution for gppsum=1 case
      do ig = 1, ngpown
        inv_igp_index(ig) = ig * ncouls / ngpown
      enddo

! Try this identity and random
      do ig = 1, ncouls
        indinv(ig) = ig
      enddo

      tol = 1D-6
      gamma = 0.5d0
      sexcut=4.0d0
      limitone=1D0/(tol*4D0)
      limittwo=0.5d0**2


!      call mpi_barrier(mpi_comm_world,ierr)
      call timget(starttime)

      do n1=1,number_bands

! n1true = "True" band index of the band n1 w.r.t. all bands

        !!! n1true = peinf%indext_dist(n1,ipe)

! energy of the |n1,k-q> state

        e_n1kq = ekq(n1,1)

! occupation of the |n1,k-q> state

        flag_occ = (n1.le.nvband)

        !!!tempval=abs(e_n1kq-sig%efermi)
        !!!if (tempval .lt. tol) then
        !!!  occ = 0.5d0 ! Fermi-Dirac distribution = 1/2 at Fermi level
        !!!else
          occ = 1.0d0
        !!!endif

! JRD: compute the static CH for the static remainder

        call timget(starttime_stat)

        !!!if (sig%exact_ch.eq.1) then
!$OMP PARALLEL do private (my_igp,igp,indigp,igmax,mygpvar1,mygpvar2,ig,schs, &
!$OMP                       matngmatmgp,matngpmatmg,schstemp) reduction(+:achstemp) &
!$OMP          schedule(dynamic)
          do my_igp = 1, ngpown

            ! This index could be bad

            indigp = inv_igp_index(my_igp)
            igp = indinv(indigp)

            if (igp .gt. ncouls .or. igp .le. 0) cycle

              igmax=ncouls
          

            mygpvar1 = CONJG(aqsmtemp(igp,n1))
            mygpvar2 = aqsntemp(igp,n1)

            schstemp = 0D0

! We do two loops here for performance. Don`t want to evaluate if statements inside loop
! at every iteration

              do ig = 1, igmax
                !schs=-I_eps_array(ig,my_igp)
! JRD: Cycle bad for vectorization.
! I_eps_array is already set to zero above for these ig,igp
!             if (abs(schs).lt.tol) cycle
                !matngmatmgp = aqsntemp(ig,n1) * mygpvar1
                schstemp = schstemp - aqsntemp(ig,n1) * I_eps_array(ig,my_igp) * mygpvar1
                !schstemp = schstemp + matngmatmgp * schs
              enddo

            achstemp = achstemp + schstemp*vcoul(igp)*0.5d0
          enddo
!$OMP END PARALLEL DO
        !!!endif ! sig%exact_ch.eq.1

        call timget(endtime_stat)

        time_stat = time_stat + endtime_stat - starttime_stat

        do iw=nstart,nend
          wx_array(iw) = e_lk - e_n1kq + dw*(iw-2)
          if (abs(wx_array(iw)) .lt. tol) wx_array(iw) = tol
        enddo

        call timget(starttime_dyn)

! JRD: This Loop is Performance critical. Make Sure you don`t mess it up

        ALLOCATE(ssx_array(3))
        ALLOCATE(sch_array(3))
        ALLOCATE(ssxa(ncouls))
        ALLOCATE(scha(ncouls))

!$OMP PARALLEL private (my_igp,igp,indigp,igmax,mygpvar1,mygpvar2,ssx_array,sch_array,ig, &
!$OMP                      wtilde,wtilde2,halfinvwtilde,ssxcutoff,matngmatmgp,matngpmatmg,sch,ssx, &
!$OMP                      iw,delw,delw2,Omega2,scht,ssxt,wxt, &
!$OMP                      rden,cden,ssxa,scha,delwr,wdiffr,occfact,igend,igbeg)

!$OMP DO reduction(+:asxtemp,acht_n1_loc,achtemp) schedule(dynamic)
        do my_igp = 1, ngpown

          indigp = inv_igp_index(my_igp)
          igp = indinv(indigp)

          if (igp .gt. ncouls .or. igp .le. 0) cycle

        igmax=ncouls
          ssx_array = 0D0
          sch_array = 0D0

          mygpvar1 = CONJG(aqsmtemp(igp,n1))
          mygpvar2 = aqsntemp(igp,n1)

          if (flag_occ) then

            do iw=nstart,nend

              scht=0D0
              ssxt=0D0
              wxt = wx_array(iw)

                do ig = 1, igmax

                  wtilde = wtilde_array(ig,my_igp)
                  wtilde2 = wtilde**2
                  Omega2 = wtilde2 * I_eps_array(ig,my_igp)

! Cycle bad for vectorization. Not needed wtilde is zero
                  !if (abs(Omega2) .lt. tol) cycle

                  matngmatmgp = aqsntemp(ig,n1) * mygpvar1

                  wdiff = wxt - wtilde

                  cden = wdiff
                  rden = cden * CONJG(cden)
                  rden = 1D0 / rden
                  delw = wtilde * CONJG(cden) * rden
                  delwr = delw*CONJG(delw)
                  wdiffr = wdiff*CONJG(wdiff)

! This Practice is bad for vectorization and understanding of the output.
! JRD: Complex division is hard to vectorize. So, we help the compiler.
                  if (wdiffr.gt.limittwo .and. delwr.lt.limitone) then
                    sch = delw * I_eps_array(ig,my_igp)
                    cden = wxt**2 - wtilde2
                    rden = cden*CONJG(cden)
                    rden = 1D0 / rden
                    ssx = Omega2 * CONJG(cden) * rden
                  else if ( delwr .gt. tol) then
                    sch = 0.0d0
                    cden = (4.0d0 * wtilde2 * (delw + 0.5D0 ))
                    rden = cden*CONJG(cden)
                    rden = 1D0 / rden
                    ssx = -Omega2 * CONJG(cden) * rden * delw
                  else
                    sch = 0.0d0
                    ssx = 0.0d0
                  endif

! JRD: Breaks vectorization. But, I will have to fix later because
! leaving it out breaks GSM example.
                  ssxcutoff = sexcut*abs(I_eps_array(ig,my_igp))
                  if (abs(ssx) .gt. ssxcutoff .and. wxt .lt. 0.0d0) ssx=0.0d0

                  ssxa(ig) = matngmatmgp*ssx
                  scha(ig) = matngmatmgp*sch

                  ssxt = ssxt + ssxa(ig)
                  scht = scht + scha(ig)

                enddo ! loop over g

              ssx_array(iw) = ssx_array(iw) + ssxt
              sch_array(iw) = sch_array(iw) + 0.5D0*scht

            enddo

          else
!            call timget(starttime_noFlagOCC)

            do igbeg = 1,igmax,igblk
            igend = min(igbeg+igblk-1,igmax)
            do iw=nstart,nend

              scht=0D0
              ssxt=0D0
              wxt = wx_array(iw)


! !dir$ no unroll
                do ig = igbeg, min(igend,igmax)
                  wdiff = wxt - wtilde_array(ig,my_igp)

                  cden = wdiff
                  rden = cden * CONJG(cden)
                  rden = 1D0 / rden
                  delw = wtilde_array(ig,my_igp) * CONJG(cden) * rden
                  delwr = delw*CONJG(delw)
                  wdiffr = wdiff*CONJG(wdiff)

! JRD: Complex division is hard to vectorize. So, we help the compiler.
                  scha(ig) = mygpvar1 * aqsntemp(ig,n1) * delw * I_eps_array(ig,my_igp)
!                   scha_temp = mygpvar1 * aqsntemp(ig,n1) * delw * I_eps_array(ig,my_igp)

! JRD: This if is OK for vectorization
                   if (wdiffr.gt.limittwo .and. delwr.lt.limitone) then
                     scht = scht + scha(ig)
                   endif
                enddo ! loop over g
!                do ig = igbeg, min(igend,igmax)
!                     scht = scht + scha(ig)
!                     enddo

!                do ig = igbeg, min(igend,igmax)
!                     scht = scht + scha(ig)
!                 enddo

              sch_array(iw) = sch_array(iw) + 0.5D0*scht

!-----------------------
! JRD: Compute GPP Error...
! GPP Model Error Estimate

            enddo
            enddo
!            call timget(endtime_noFlagOCC)
!            totaltime_noFlagOCC = totaltime_noFlagOCC + (endtime_noFlagOCC - starttime_noFlagOCC)

          endif

! If a valence band, then accumulate SX contribution.

          if (flag_occ) then
            do iw=nstart,nend
              asxtemp(iw) = asxtemp(iw) - ssx_array(iw) * occ * vcoul(igp)
            enddo
          endif

          do iw=nstart,nend

            achtemp(iw) = achtemp(iw) + sch_array(iw) * vcoul(igp)
          enddo

! Logging CH convergence.

          acht_n1_loc(n1) = acht_n1_loc(n1) + sch_array(2) * vcoul(igp)

        enddo ! igp
!$OMP END DO

!$OMP END PARALLEL

        DEALLOCATE(ssx_array)
        DEALLOCATE(sch_array)
        DEALLOCATE(ssxa)
        DEALLOCATE(scha)

        call timget(endtime_dyn)
        time_dyn = time_dyn + endtime_dyn - starttime_dyn

      enddo ! over ipe bands (n1)
!      call mpi_barrier(mpi_comm_world,ierr)

      call timget(endtime)

      DEALLOCATE(vcoul)
      DEALLOCATE(aqsntemp)
      DEALLOCATE(aqsmtemp)
      DEALLOCATE(inv_igp_index)
      DEALLOCATE(indinv)
      DEALLOCATE(I_eps_array)
      DEALLOCATE(acht_n1_loc)
      DEALLOCATE(wtilde_array)
      DEALLOCATE(ekq)

!      if(mype==0)then
!        write(6,*) "noflag_occ time:",totaltime_noFlagOCC 
        write(6,*) "Runtime:", endtime-starttime
        write(6,*) "Runtime Stat:", time_stat
        write(6,*) "Runtime Dyn:", time_dyn
        write(6,*) "Answer[1]:",achtemp(1)
        write(6,*) "Answer[2]:",achtemp(2)
        write(6,*) "Answer[3]:",achtemp(3)
!      endif

      DEALLOCATE(achtemp)
      DEALLOCATE(asxtemp)
!      call mpi_finalize(ierr)

end program

subroutine timget(wall)
  real(kind(1.0d0)) :: wall

  integer :: values(8)

  call date_and_time(VALUES=values)
  wall=((values(3)*24.0d0+values(5))*60.0d0 &
    +values(6))*60.0d0+values(7)+values(8)*1.0d-3
  return
end subroutine timget

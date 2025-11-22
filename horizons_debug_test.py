from astroquery.jplhorizons import Horizons

try:
    obj = Horizons(id='1P/Halley', id_type='designation', location='500')
    eph = obj.ephemerides()
    print("SUCCESS: HALLEY returned", eph[0])
except Exception as e:
    print("FAILURE:", e)
